use crate::{
    io::{
        AssetReaderError, ErasedAssetReader, MissingAssetSourceError,
        MissingProcessedAssetReaderError, Reader,
    },
    loader_builders::NestedLoadBuilder,
    meta::{AssetHash, AssetMeta, AssetMetaDyn, ProcessedInfo, ProcessedInfoMinimal, Settings},
    path::AssetPath,
    Asset, AssetIndex, AssetLoadError, AssetServer, AssetServerMode, Assets, ErasedAssetIndex,
    Handle, UntypedAssetId, UntypedHandle,
};
use alloc::{boxed::Box, string::ToString, vec::Vec};
use atomicow::CowArc;
use bevy_ecs::{error::BevyError, world::World};
use bevy_platform::{
    cell::SyncCell,
    collections::{hash_map::Entry, HashMap, HashSet},
};
use bevy_reflect::TypePath;
use bevy_tasks::{BoxedFuture, ConditionalSendFuture};
use core::{
    any::{type_name, Any, TypeId},
    convert::Infallible,
};
use downcast_rs::{impl_downcast, Downcast};
use ron::error::SpannedError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;
use tracing::error;

/// Loads an [`Asset`] from a given byte [`Reader`]. This can accept [`AssetLoader::Settings`], which configure how the [`Asset`]
/// should be loaded.
///
/// This trait is generally used in concert with [`AssetReader`](crate::io::AssetReader) to load assets from a byte source.
///
/// For a complementary version of this trait that can save assets, see [`AssetSaver`](crate::saver::AssetSaver).
pub trait AssetLoader: TypePath + Send + Sync + 'static {
    /// The top level [`Asset`] loaded by this [`AssetLoader`].
    type Asset: Asset;
    /// The settings type used by this [`AssetLoader`].
    type Settings: Settings + Default + Serialize + for<'a> Deserialize<'a>;
    /// The type of [error](`std::error::Error`) which could be encountered by this loader.
    type Error: Into<BevyError>;
    /// Asynchronously loads [`AssetLoader::Asset`] (and any other labeled assets) from the bytes provided by [`Reader`].
    fn load(
        &self,
        reader: &mut dyn Reader,
        settings: &Self::Settings,
        load_context: &mut LoadContext,
    ) -> impl ConditionalSendFuture<Output = Result<Self::Asset, Self::Error>>;

    /// Returns a list of extensions supported by this [`AssetLoader`], without the preceding dot.
    /// Note that users of this [`AssetLoader`] may choose to load files with a non-matching extension.
    fn extensions(&self) -> &[&str] {
        &[]
    }
}

/// Provides type-erased access to an [`AssetLoader`].
pub trait ErasedAssetLoader: Send + Sync + 'static {
    /// Asynchronously loads the asset(s) from the bytes provided by [`Reader`].
    fn load<'a>(
        &'a self,
        reader: &'a mut dyn Reader,
        settings: &'a dyn Settings,
        load_context: LoadContext<'a>,
    ) -> BoxedFuture<'a, Result<ErasedLoadedAsset, BevyError>>;

    /// Returns a list of extensions supported by this asset loader, without the preceding dot.
    fn extensions(&self) -> &[&str];
    /// Deserializes metadata from the input `meta` bytes into the appropriate type (erased as [`Box<dyn AssetMetaDyn>`]).
    fn deserialize_meta(&self, meta: &[u8]) -> Result<Box<dyn AssetMetaDyn>, DeserializeMetaError>;
    /// Returns the default meta value for the [`AssetLoader`] (erased as [`Box<dyn AssetMetaDyn>`]).
    fn default_meta(&self) -> Box<dyn AssetMetaDyn>;
    /// Returns the type path of the [`AssetLoader`].
    fn type_path(&self) -> &'static str;
    /// Returns the [`TypeId`] of the [`AssetLoader`].
    fn type_id(&self) -> TypeId;
    /// Returns the type name of the top-level [`Asset`] loaded by the [`AssetLoader`].
    fn asset_type_name(&self) -> &'static str;
    /// Returns the [`TypeId`] of the top-level [`Asset`] loaded by the [`AssetLoader`].
    fn asset_type_id(&self) -> TypeId;
}

impl<L> ErasedAssetLoader for L
where
    L: AssetLoader + Send + Sync,
{
    /// Processes the asset in an asynchronous closure.
    fn load<'a>(
        &'a self,
        reader: &'a mut dyn Reader,
        settings: &'a dyn Settings,
        mut load_context: LoadContext<'a>,
    ) -> BoxedFuture<'a, Result<ErasedLoadedAsset, BevyError>> {
        Box::pin(async move {
            let settings = settings
                .downcast_ref::<L::Settings>()
                .expect("AssetLoader settings should match the loader type");
            let asset = <L as AssetLoader>::load(self, reader, settings, &mut load_context)
                .await
                .map_err(Into::into)?;
            Ok(load_context.finish(asset).into())
        })
    }

    fn extensions(&self) -> &[&str] {
        <L as AssetLoader>::extensions(self)
    }

    fn deserialize_meta(&self, meta: &[u8]) -> Result<Box<dyn AssetMetaDyn>, DeserializeMetaError> {
        let meta = AssetMeta::<L, ()>::deserialize(meta)?;
        Ok(Box::new(meta))
    }

    fn default_meta(&self) -> Box<dyn AssetMetaDyn> {
        Box::new(AssetMeta::<L, ()>::new(crate::meta::AssetAction::Load {
            loader: self.type_path().to_string(),
            settings: L::Settings::default(),
        }))
    }

    fn type_path(&self) -> &'static str {
        L::type_path()
    }

    fn type_id(&self) -> TypeId {
        TypeId::of::<L>()
    }

    fn asset_type_name(&self) -> &'static str {
        type_name::<L::Asset>()
    }

    fn asset_type_id(&self) -> TypeId {
        TypeId::of::<L::Asset>()
    }
}

pub(crate) struct LabeledAsset {
    pub(crate) asset: ErasedLoadedAsset,
    pub(crate) handle: UntypedHandle,
}

/// Produces a deferred labeled asset, given the server to build its [`LoadContext`] from.
pub(crate) type DeferredLoad = Box<
    dyn FnOnce(AssetServer) -> BoxedFuture<'static, Result<ErasedLoadedAsset, BevyError>> + Send,
>;

/// When a labeled asset added with [`LoadContext::defer_labeled_asset`] starts loading.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum DeferredLoadMode {
    /// Starts as soon as the asset that added it is loaded.
    #[default]
    Eager,
    /// Starts when it is requested through the [`AssetServer`], for example with
    /// [`AssetServer::load`] and its labeled path. Holding its handle does not start it.
    ///
    /// Until it is requested, it is [`LoadState::NotLoaded`](crate::LoadState::NotLoaded) and
    /// assets that depend on it do not wait for it.
    OnDemand,
}

/// A labeled asset registered with [`LoadContext::defer_labeled_asset`], whose value has not been
/// produced yet.
pub(crate) struct DeferredLabeledAsset {
    pub(crate) label: CowArc<'static, str>,
    pub(crate) handle: UntypedHandle,
    pub(crate) asset_type_name: &'static str,
    pub(crate) mode: DeferredLoadMode,
    pub(crate) load: SyncCell<DeferredLoad>,
}

/// Adds `asset` to a set of labeled assets, replacing any asset with the same label.
fn insert_labeled_asset(
    labeled_assets: &mut Vec<LabeledAsset>,
    label_to_asset_index: &mut HashMap<CowArc<'static, str>, usize>,
    asset_id_to_asset_index: &mut HashMap<UntypedAssetId, usize>,
    label: CowArc<'static, str>,
    asset: LabeledAsset,
) {
    match label_to_asset_index.entry(label) {
        Entry::Occupied(entry) => {
            // TODO: We should probably treat this as an error. It seems unlikely someone wants
            // to replace a subasset - this is probably accidental.
            let index = *entry.get();
            // Note: we don't need to mess with the `asset_id_to_asset_index` here, since we
            // know the same path to `get_or_create_path_handle` will return the same handle as
            // long as the handle remains alive, and we hold the handle in `LabeledAsset`.
            labeled_assets[index] = asset;
        }
        Entry::Vacant(entry) => {
            entry.insert(labeled_assets.len());
            asset_id_to_asset_index.insert(asset.handle.id(), labeled_assets.len());
            labeled_assets.push(asset);
        }
    }
}

/// The successful result of an [`AssetLoader::load`] call. This contains the loaded "root" asset and any other "labeled" assets produced
/// by the loader. It also holds the input [`AssetMeta`] (if it exists) and tracks dependencies:
/// * normal dependencies: dependencies that must be loaded as part of this asset load (ex: assets a given asset has handles to).
/// * Loader dependencies: dependencies whose actual asset values are used during the load process
pub struct LoadedAsset<A: Asset> {
    pub(crate) value: A,
    pub(crate) dependencies: HashSet<ErasedAssetIndex>,
    pub(crate) loader_dependencies: HashMap<AssetPath<'static>, AssetHash>,
    /// The subassets of this asset.
    pub(crate) labeled_assets: Vec<LabeledAsset>,
    /// The mapping from subasset labels to their index in [`Self::labeled_assets`].
    pub(crate) label_to_asset_index: HashMap<CowArc<'static, str>, usize>,
    /// The mapping from a subasset asset IDs to their index in [`Self::labeled_assets`].
    ///
    /// This is entirely redundant with [`Self::labeled_assets`], but it allows looking up the
    /// labeled asset by its asset ID.
    pub(crate) asset_id_to_asset_index: HashMap<UntypedAssetId, usize>,
    /// The subassets of this asset that are produced after it is committed.
    pub(crate) deferred_labeled_assets: Vec<DeferredLabeledAsset>,
}

impl<A: Asset> LoadedAsset<A> {
    /// Create a new loaded asset. This will use [`VisitAssetDependencies`](crate::VisitAssetDependencies) to populate `dependencies`.
    pub fn new_with_dependencies(value: A) -> Self {
        let mut dependencies = <HashSet<_>>::default();
        value.visit_dependencies(&mut |id| {
            let Ok(asset_index) = id.try_into() else {
                return;
            };
            dependencies.insert(asset_index);
        });
        LoadedAsset {
            value,
            dependencies,
            loader_dependencies: HashMap::default(),
            labeled_assets: Default::default(),
            label_to_asset_index: Default::default(),
            asset_id_to_asset_index: Default::default(),
            deferred_labeled_assets: Vec::new(),
        }
    }

    /// Cast (and take ownership) of the [`Asset`] value of the given type.
    pub fn take(self) -> A {
        self.value
    }

    /// Retrieves a reference to the internal [`Asset`] type.
    pub fn get(&self) -> &A {
        &self.value
    }

    /// Returns the [`ErasedLoadedAsset`] for the given label, if it exists.
    pub fn get_labeled(&self, label: impl AsRef<str>) -> Option<&ErasedLoadedAsset> {
        self.label_to_asset_index
            .get(label.as_ref())
            .map(|index| self.labeled_assets.get(*index).unwrap())
            .map(|a| &a.asset)
    }

    /// Returns the labeled asset given its asset ID if it exists.
    ///
    /// This can be used to get the asset from its handle since `&Handle` implements
    /// [`Into<UntypedAssetId>`].
    pub fn get_labeled_by_id(&self, id: impl Into<UntypedAssetId>) -> Option<&ErasedLoadedAsset> {
        let index = self.asset_id_to_asset_index.get(&id.into())?;
        let labeled = &self.labeled_assets[*index];
        Some(&labeled.asset)
    }

    /// Iterate over all labels for "labeled assets" in the loaded asset
    pub fn iter_labels(&self) -> impl Iterator<Item = &str> {
        self.label_to_asset_index.keys().map(|s| &**s)
    }
}

impl<A: Asset> From<A> for LoadedAsset<A> {
    fn from(asset: A) -> Self {
        LoadedAsset::new_with_dependencies(asset)
    }
}

/// A "type erased / boxed" counterpart to [`LoadedAsset`]. This is used in places where the loaded type is not statically known.
pub struct ErasedLoadedAsset {
    pub(crate) value: Box<dyn AssetContainer>,
    pub(crate) dependencies: HashSet<ErasedAssetIndex>,
    pub(crate) loader_dependencies: HashMap<AssetPath<'static>, AssetHash>,
    /// The subassets of this asset.
    pub(crate) labeled_assets: Vec<LabeledAsset>,
    /// The mapping from subasset labels to their index in [`Self::labeled_assets`].
    pub(crate) label_to_asset_index: HashMap<CowArc<'static, str>, usize>,
    /// The mapping from subasset asset IDs to their index in [`Self::labeled_assets`].
    ///
    /// This is entirely redundant with [`Self::labeled_assets`], but it allows looking up the
    /// labeled asset by its asset ID.
    pub(crate) asset_id_to_asset_index: HashMap<UntypedAssetId, usize>,
    /// The subassets of this asset that are produced after it is committed.
    pub(crate) deferred_labeled_assets: Vec<DeferredLabeledAsset>,
}

impl<A: Asset> From<LoadedAsset<A>> for ErasedLoadedAsset {
    fn from(asset: LoadedAsset<A>) -> Self {
        ErasedLoadedAsset {
            value: Box::new(asset.value),
            dependencies: asset.dependencies,
            loader_dependencies: asset.loader_dependencies,
            labeled_assets: asset.labeled_assets,
            label_to_asset_index: asset.label_to_asset_index,
            asset_id_to_asset_index: asset.asset_id_to_asset_index,
            deferred_labeled_assets: asset.deferred_labeled_assets,
        }
    }
}

impl ErasedLoadedAsset {
    /// Cast (and take ownership) of the [`Asset`] value of the given type. This will return [`Some`] if
    /// the stored type matches `A` and [`None`] if it does not.
    pub fn take<A: Asset>(self) -> Option<A> {
        self.value.downcast::<A>().map(|a| *a).ok()
    }

    /// Retrieves a reference to the internal [`Asset`] type, if it matches the type `A`. Otherwise returns [`None`].
    pub fn get<A: Asset>(&self) -> Option<&A> {
        self.value.downcast_ref::<A>()
    }

    /// Retrieves the [`TypeId`] of the stored [`Asset`] type.
    pub fn asset_type_id(&self) -> TypeId {
        (*self.value).type_id()
    }

    /// Retrieves the `type_name` of the stored [`Asset`] type.
    pub fn asset_type_name(&self) -> &'static str {
        self.value.asset_type_name()
    }

    /// Returns the [`ErasedLoadedAsset`] for the given label, if it exists.
    pub fn get_labeled(&self, label: impl AsRef<str>) -> Option<&ErasedLoadedAsset> {
        self.label_to_asset_index
            .get(label.as_ref())
            .map(|index| self.labeled_assets.get(*index).unwrap())
            .map(|a| &a.asset)
    }

    /// Returns the labeled asset given its asset ID if it exists.
    ///
    /// This can be used to get the asset from its handle since `&Handle` implements
    /// [`Into<UntypedAssetId>`].
    pub fn get_labeled_by_id(&self, id: impl Into<UntypedAssetId>) -> Option<&ErasedLoadedAsset> {
        let index = self.asset_id_to_asset_index.get(&id.into())?;
        let labeled = &self.labeled_assets[*index];
        Some(&labeled.asset)
    }

    /// Iterate over all labels for "labeled assets" in the loaded asset
    pub fn iter_labels(&self) -> impl Iterator<Item = &str> {
        self.label_to_asset_index.keys().map(|s| &**s)
    }

    /// Cast this loaded asset as the given type. If the type does not match,
    /// the original type-erased asset is returned.
    #[cfg_attr(
        not(target_arch = "wasm32"),
        expect(
            clippy::result_large_err,
            reason = "Returning the passed in ErasedLoadedAsset"
        )
    )]
    pub fn downcast<A: Asset>(mut self) -> Result<LoadedAsset<A>, ErasedLoadedAsset> {
        match self.value.downcast::<A>() {
            Ok(value) => Ok(LoadedAsset {
                value: *value,
                dependencies: self.dependencies,
                loader_dependencies: self.loader_dependencies,
                labeled_assets: self.labeled_assets,
                label_to_asset_index: self.label_to_asset_index,
                asset_id_to_asset_index: self.asset_id_to_asset_index,
                deferred_labeled_assets: self.deferred_labeled_assets,
            }),
            Err(value) => {
                self.value = value;
                Err(self)
            }
        }
    }

    /// Runs every deferred labeled asset to completion, adding each as an ordinary labeled asset
    /// so the result is complete. Used where the loaded value is consumed directly instead of
    /// being committed to the [`World`], such as nested immediate loads and asset processing.
    pub(crate) async fn load_deferred_labeled_assets(
        &mut self,
        asset_server: &AssetServer,
    ) -> Result<(), BevyError> {
        // Deferred assets may defer more labeled assets of their own.
        while !self.deferred_labeled_assets.is_empty() {
            for deferred in core::mem::take(&mut self.deferred_labeled_assets) {
                let load = SyncCell::to_inner(deferred.load);
                let mut asset = load(asset_server.clone()).await?;
                self.deferred_labeled_assets
                    .append(&mut asset.deferred_labeled_assets);
                // The deferred asset's reads belong to this load, since that is what gets
                // processed and hot-reloaded.
                self.loader_dependencies.extend(
                    asset
                        .loader_dependencies
                        .iter()
                        .map(|(path, hash)| (path.clone(), *hash)),
                );
                insert_labeled_asset(
                    &mut self.labeled_assets,
                    &mut self.label_to_asset_index,
                    &mut self.asset_id_to_asset_index,
                    deferred.label,
                    LabeledAsset {
                        asset,
                        handle: deferred.handle,
                    },
                );
            }
        }
        Ok(())
    }
}

/// A type erased container for an [`Asset`] value that is capable of inserting the [`Asset`] into a [`World`]'s [`Assets`] collection.
pub(crate) trait AssetContainer: Downcast + Any + Send + Sync + 'static {
    fn insert(self: Box<Self>, id: AssetIndex, world: &mut World);
    fn asset_type_name(&self) -> &'static str;
}

impl_downcast!(AssetContainer);

impl<A: Asset> AssetContainer for A {
    fn insert(self: Box<Self>, index: AssetIndex, world: &mut World) {
        // We only ever call this if we know the asset is still alive, so it is fine to unwrap here.
        world
            .resource_mut::<Assets<A>>()
            .insert(index, *self)
            .expect("the AssetIndex is still valid");
    }

    fn asset_type_name(&self) -> &'static str {
        type_name::<A>()
    }
}

/// An error that occurs when attempting an async load using [`NestedLoadBuilder`].
#[derive(Error, Debug)]
pub enum LoadDirectError {
    /// The asset path was empty.
    #[error("Attempted to load an asset with an empty path \"{0}\"")]
    EmptyPath(AssetPath<'static>),
    /// Loading an asset path with a subasset at the end is unsupported. See issue [#18291].
    ///
    /// [#18291]: https://github.com/bevyengine/bevy/issues/18291
    #[error("Requested to load an asset path ({0:?}) with a subasset, but this is unsupported. See issue #18291")]
    RequestedSubasset(AssetPath<'static>),
    /// A general [`AssetLoadError`] for an asset dependency.
    #[error("Failed to load dependency {dependency:?} {error}")]
    LoadError {
        /// Which dependency failed.
        dependency: AssetPath<'static>,
        /// The original error for that dependency.
        error: Box<AssetLoadError>,
    },
}

/// An error that occurs while deserializing [`AssetMeta`].
#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum DeserializeMetaError {
    /// Failed to deserialize the asset metadata.
    #[error("Failed to deserialize asset meta: {0:?}")]
    DeserializeSettings(#[from] SpannedError),
    /// Failed to deserialize the minimal asset metadata.
    #[error("Failed to deserialize minimal asset meta: {0:?}")]
    DeserializeMinimal(SpannedError),
}

/// A context that provides access to assets in [`AssetLoader`]s, tracks dependencies, and collects asset load state.
///
/// Any asset state accessed by [`LoadContext`] will be tracked and stored for use in dependency events and asset preprocessing.
pub struct LoadContext<'a> {
    pub(crate) asset_server: &'a AssetServer,
    /// Specifies whether dependencies that are loaded deferred should be loaded.
    ///
    /// This allows us to skip loads for cases where we're never going to use the asset and we just
    /// need the dependency information, for example during asset processing.
    pub(crate) should_load_dependencies: bool,
    populate_hashes: bool,
    asset_path: AssetPath<'static>,
    pub(crate) dependencies: HashSet<ErasedAssetIndex>,
    /// Direct dependencies used by this loader.
    pub(crate) loader_dependencies: HashMap<AssetPath<'static>, AssetHash>,
    /// Stores the subassets added to this context.
    pub(crate) labeled_assets: Vec<LabeledAsset>,
    /// Maps the label of a subasset to the index into [`Self::labeled_assets`].
    pub(crate) label_to_asset_index: HashMap<CowArc<'static, str>, usize>,
    /// Maps the subasset asset ID to the index into [`Self::labeled_assets`].
    ///
    /// This is entirely redundant with [`Self::labeled_assets`], but it allows looking up the
    /// labeled asset by its asset ID.
    pub(crate) asset_id_to_asset_index: HashMap<UntypedAssetId, usize>,
    /// Stores the subassets registered with [`Self::defer_labeled_asset`].
    pub(crate) deferred_labeled_assets: Vec<DeferredLabeledAsset>,
}

impl<'a> LoadContext<'a> {
    /// Creates a new [`LoadContext`] instance.
    pub(crate) fn new(
        asset_server: &'a AssetServer,
        asset_path: AssetPath<'static>,
        should_load_dependencies: bool,
        populate_hashes: bool,
    ) -> Self {
        Self {
            asset_server,
            asset_path,
            populate_hashes,
            should_load_dependencies,
            dependencies: HashSet::default(),
            loader_dependencies: HashMap::default(),
            labeled_assets: Default::default(),
            label_to_asset_index: Default::default(),
            asset_id_to_asset_index: Default::default(),
            deferred_labeled_assets: Vec::new(),
        }
    }

    /// Begins a new labeled asset load. Use the returned [`LoadContext`] to load
    /// dependencies for the new asset and call [`LoadContext::finish`] to finalize the asset load.
    /// When finished, make sure you call [`LoadContext::add_loaded_labeled_asset`] to add the results back to the parent
    /// context.
    /// Prefer [`LoadContext::labeled_asset_scope`] when possible, which will automatically add
    /// the labeled [`LoadContext`] back to the parent context.
    /// [`LoadContext::begin_labeled_asset`] exists largely to enable parallel asset loading.
    ///
    /// See [`AssetPath`] for more on labeled assets.
    ///
    /// ```no_run
    /// # use bevy_asset::{Asset, LoadContext};
    /// # use bevy_reflect::TypePath;
    /// # #[derive(Asset, TypePath, Default)]
    /// # struct Image;
    /// # let load_context: LoadContext = panic!();
    /// let mut handles = Vec::new();
    /// for i in 0..2 {
    ///     let labeled = load_context.begin_labeled_asset();
    ///     handles.push(std::thread::spawn(move || {
    ///         (i.to_string(), labeled.finish(Image::default()))
    ///     }));
    /// }
    ///
    /// for handle in handles {
    ///     let (label, loaded_asset) = handle.join().unwrap();
    ///     load_context.add_loaded_labeled_asset(label, loaded_asset);
    /// }
    /// ```
    pub fn begin_labeled_asset(&self) -> LoadContext<'_> {
        LoadContext::new(
            self.asset_server,
            self.asset_path.clone(),
            self.should_load_dependencies,
            self.populate_hashes,
        )
    }

    /// Creates a new [`LoadContext`] for the given `label`. The `load` function is responsible for loading an [`Asset`] of
    /// type `A`. `load` will be called immediately and the result will be used to finalize the [`LoadContext`], resulting in a new
    /// [`LoadedAsset`], which is registered under the `label` label.
    ///
    /// This exists to remove the need to manually call [`LoadContext::begin_labeled_asset`] and then manually register the
    /// result with [`LoadContext::add_loaded_labeled_asset`].
    ///
    /// See [`AssetPath`] for more on labeled assets.
    pub fn labeled_asset_scope<A: Asset, E>(
        &mut self,
        label: impl Into<CowArc<'static, str>>,
        load: impl FnOnce(&mut LoadContext) -> Result<A, E>,
    ) -> Result<Handle<A>, E> {
        let mut context = self.begin_labeled_asset();
        let asset = load(&mut context)?;
        let loaded_asset = context.finish(asset);
        Ok(self.add_loaded_labeled_asset(label, loaded_asset))
    }

    /// This will add the given `asset` as a "labeled [`Asset`]" with the `label` label.
    ///
    /// # Warning
    ///
    /// This will not assign dependencies to the given `asset`. If adding an asset
    /// with dependencies generated from calls such as [`LoadContext::load`], use
    /// [`LoadContext::labeled_asset_scope`] or [`LoadContext::begin_labeled_asset`] to generate a
    /// new [`LoadContext`] to track the dependencies for the labeled asset.
    ///
    /// See [`AssetPath`] for more on labeled assets.
    pub fn add_labeled_asset<A: Asset>(
        &mut self,
        label: impl Into<CowArc<'static, str>>,
        asset: A,
    ) -> Handle<A> {
        let Ok(handle) = self.labeled_asset_scope(label, |_| Ok::<_, Infallible>(asset));
        handle
    }

    /// Add a [`LoadedAsset`] that is a "labeled sub asset" of the root path of this load context.
    /// This can be used in combination with [`LoadContext::begin_labeled_asset`] to parallelize
    /// sub asset loading.
    ///
    /// See [`AssetPath`] for more on labeled assets.
    pub fn add_loaded_labeled_asset<A: Asset>(
        &mut self,
        label: impl Into<CowArc<'static, str>>,
        loaded_asset: LoadedAsset<A>,
    ) -> Handle<A> {
        self.add_erased_loaded_labeled_asset_internal(
            label.into(),
            loaded_asset.into(),
            Some(type_name::<A>()),
        )
        .typed_debug_checked()
    }

    /// Adds an [`ErasedLoadedAsset`] that is a "labeled sub asset" of the root path of this load
    /// context. This can be used in combination with [`LoadContext::begin_labeled_asset`] to
    /// parallelize sub asset loading.
    ///
    /// See [`AssetPath`] for more on labeled assets.
    pub fn add_erased_loaded_labeled_asset(
        &mut self,
        label: impl Into<CowArc<'static, str>>,
        loaded_asset: ErasedLoadedAsset,
    ) -> UntypedHandle {
        self.add_erased_loaded_labeled_asset_internal(label.into(), loaded_asset, None)
    }

    /// Internal versions of the above that are type erased, but still may include `type_name` for
    /// debug information.
    fn add_erased_loaded_labeled_asset_internal(
        &mut self,
        label: CowArc<'static, str>,
        mut loaded_asset: ErasedLoadedAsset,
        type_name: Option<&str>,
    ) -> UntypedHandle {
        let labeled_path = self.asset_path.clone().with_label(label.clone());
        let handle = self.asset_server.get_or_create_path_handle_erased(
            labeled_path,
            loaded_asset.asset_type_id(),
            type_name,
            None,
        );
        // Labels are flat, so a labeled asset's deferred labeled assets belong to this context.
        self.deferred_labeled_assets
            .append(&mut loaded_asset.deferred_labeled_assets);
        insert_labeled_asset(
            &mut self.labeled_assets,
            &mut self.label_to_asset_index,
            &mut self.asset_id_to_asset_index,
            label,
            LabeledAsset {
                asset: loaded_asset,
                handle: handle.clone(),
            },
        );
        handle
    }

    /// Adds a labeled asset whose value is produced after [`AssetLoader::load`] returns, and
    /// returns its handle.
    ///
    /// The asset this context is loading is committed without waiting for `load`. `load` runs
    /// later on the [`IoTaskPool`](bevy_tasks::IoTaskPool), like every asset load, with its own
    /// [`LoadContext`] for the same asset path, and its asset is committed like any other labeled
    /// asset once it finishes. Use this for labeled assets that are expensive to produce and
    /// useful on their own, such as the meshes of a large scene, so they arrive as they finish
    /// instead of all at once. `mode` decides when `load` starts:
    ///
    /// - [`DeferredLoadMode::Eager`]: once the asset this context is loading is committed. All
    ///   eager labeled assets of a load start together, so loads requested after them queue
    ///   behind them on the task pool.
    /// - [`DeferredLoadMode::OnDemand`]: when the labeled asset is requested through the
    ///   [`AssetServer`], for example with [`AssetServer::load`] and its labeled path. Holding the
    ///   handle does not start it, so the caller decides how many are loading or resident at once.
    ///
    /// The deferred asset is [`LoadState::Loading`](crate::LoadState::Loading) from when it starts
    /// until `load` finishes. Assets that depend on it, through a `#[dependency]` handle or
    /// [`LoadContext::get_label_handle`], reach
    /// [`RecursiveDependencyLoadState::Loaded`](crate::RecursiveDependencyLoadState::Loaded) only
    /// after it does, unless it is an on-demand asset that was not requested when they were
    /// committed. If `load` fails, the deferred asset fails to load and its dependents see a
    /// dependency failure; the root asset stays loaded.
    ///
    /// Only handles keep the work alive. It is cancelled once every handle to the deferred asset
    /// is dropped, so store the returned handle in the root asset (or another labeled asset) if
    /// the deferred asset should stay available regardless of what else holds it. Reloading the
    /// root asset restarts its eager deferred assets and the on-demand ones that were requested.
    /// The files `load` reads are loader dependencies of the root asset.
    ///
    /// Deferral is not observable to code that consumes the loaded value directly instead of
    /// through the [`AssetServer`]: nested immediate loads and asset processing run `load` before
    /// returning, in either mode, so it behaves like [`LoadContext::labeled_asset_scope`] there.
    ///
    /// `label` must be unique among this load's labels. Labeled assets added through the
    /// deferred [`LoadContext`] are committed together with the deferred asset.
    ///
    /// ```no_run
    /// # use bevy_asset::{Asset, DeferredLoadMode, LoadContext};
    /// # use bevy_reflect::TypePath;
    /// # #[derive(Asset, TypePath)]
    /// # struct Mesh(Vec<u8>);
    /// # let load_context: &mut LoadContext = panic!();
    /// let handle =
    ///     load_context.defer_labeled_asset("Mesh0", DeferredLoadMode::Eager, |load_context| {
    ///         Box::pin(async move {
    ///             let bytes = load_context.read_asset_bytes("mesh0.bin").await?;
    ///             Ok::<_, bevy_asset::ReadAssetBytesError>(Mesh(bytes))
    ///         })
    ///     });
    /// ```
    pub fn defer_labeled_asset<A, E, F>(
        &mut self,
        label: impl Into<CowArc<'static, str>>,
        mode: DeferredLoadMode,
        load: F,
    ) -> Handle<A>
    where
        A: Asset,
        E: Into<BevyError>,
        F: for<'c, 'l> FnOnce(&'c mut LoadContext<'l>) -> BoxedFuture<'c, Result<A, E>>
            + Send
            + 'static,
    {
        let label = label.into();
        let handle = self.asset_server.get_or_create_path_handle::<A>(
            self.asset_path.clone().with_label(label.clone()),
            None,
        );
        let asset_path = self.asset_path.clone();
        let should_load_dependencies = self.should_load_dependencies;
        let populate_hashes = self.populate_hashes;
        let load: DeferredLoad = Box::new(move |asset_server: AssetServer| {
            Box::pin(async move {
                let mut load_context = LoadContext::new(
                    &asset_server,
                    asset_path,
                    should_load_dependencies,
                    populate_hashes,
                );
                let asset = load(&mut load_context).await.map_err(Into::into)?;
                Ok(load_context.finish(asset).into())
            })
        });
        self.deferred_labeled_assets.push(DeferredLabeledAsset {
            label,
            handle: handle.clone().untyped(),
            asset_type_name: type_name::<A>(),
            mode,
            load: SyncCell::new(load),
        });
        handle
    }

    /// Returns `true` if an asset with the label `label` exists in this context.
    ///
    /// See [`AssetPath`] for more on labeled assets.
    pub fn has_labeled_asset<'b>(&self, label: impl Into<CowArc<'b, str>>) -> bool {
        let path = self.asset_path.clone().with_label(label.into());
        !self.asset_server.get_handles_untyped(&path).is_empty()
    }

    /// "Finishes" this context by populating the final [`Asset`] value.
    pub fn finish<A: Asset>(mut self, value: A) -> LoadedAsset<A> {
        // At this point, we assume the asset/subasset is "locked in" and won't be changed, so we
        // can ensure all the dependencies are included (in case a handle was used without loading
        // it through this `LoadContext`). If in the future we provide an API for mutating assets in
        // `LoadedAsset`, `ErasedLoadedAsset`, or `LoadContext` (for mutating existing subassets),
        // we should move this to some point after those mutations are not possible. This spot is
        // convenient because we still have access to the static type of `A`.
        value.visit_dependencies(&mut |asset_id| {
            let (type_id, index) = match asset_id {
                UntypedAssetId::Index { type_id, index } => (type_id, index),
                // UUID assets can't be loaded anyway, so just ignore this ID.
                UntypedAssetId::Uuid { .. } => return,
            };
            self.dependencies
                .insert(ErasedAssetIndex { index, type_id });
        });
        LoadedAsset {
            value,
            dependencies: self.dependencies,
            loader_dependencies: self.loader_dependencies,
            labeled_assets: self.labeled_assets,
            label_to_asset_index: self.label_to_asset_index,
            asset_id_to_asset_index: self.asset_id_to_asset_index,
            deferred_labeled_assets: self.deferred_labeled_assets,
        }
    }

    /// Gets the source asset path for this load context.
    pub fn path(&self) -> &AssetPath<'static> {
        &self.asset_path
    }

    /// Reads the asset at the given path and returns its bytes
    pub async fn read_asset_bytes<'b, 'c>(
        &'b mut self,
        path: impl Into<AssetPath<'c>>,
    ) -> Result<Vec<u8>, ReadAssetBytesError> {
        let path = path.into();
        if path.path() == Path::new("") {
            error!("Attempted to load an asset with an empty path \"{path}\"!");
            return Err(ReadAssetBytesError::EmptyPath(path.into_owned()));
        }

        let source = self.asset_server.get_source(path.source())?;
        let asset_reader = match self.asset_server.mode() {
            AssetServerMode::Unprocessed => source.reader(),
            AssetServerMode::Processed => source.processed_reader()?,
        };
        let mut reader = asset_reader.read(path.path()).await?;
        // NOTE: ensure meta is read while the asset bytes reader is still active to ensure transactionality
        // See `ProcessorGatedReader` for more info
        let hash = self.loader_dependency_hash(asset_reader, &path).await?;
        let mut bytes = Vec::new();
        reader
            .read_to_end(&mut bytes)
            .await
            .map_err(|source| ReadAssetBytesError::Io {
                path: path.path().to_path_buf(),
                source,
            })?;
        self.loader_dependencies.insert(path.clone_owned(), hash);
        Ok(bytes)
    }

    /// Records the asset at `path` as a loader dependency of this load without reading it, as
    /// [`LoadContext::read_asset_bytes`] does for the assets it reads.
    ///
    /// Changes to a loader dependency reload the asset being loaded. Use this for files the asset
    /// is built from that are read later, such as by a deferred labeled asset that may never be
    /// requested.
    pub async fn add_loader_dependency<'b>(
        &mut self,
        path: impl Into<AssetPath<'b>>,
    ) -> Result<(), ReadAssetBytesError> {
        let path = path.into();
        if path.path() == Path::new("") {
            return Err(ReadAssetBytesError::EmptyPath(path.into_owned()));
        }
        let source = self.asset_server.get_source(path.source())?;
        let asset_reader = match self.asset_server.mode() {
            AssetServerMode::Unprocessed => source.reader(),
            AssetServerMode::Processed => source.processed_reader()?,
        };
        let hash = self.loader_dependency_hash(asset_reader, &path).await?;
        self.loader_dependencies.insert(path.into_owned(), hash);
        Ok(())
    }

    /// The hash to record for loader dependency `path`, which is only known for processed assets.
    async fn loader_dependency_hash(
        &self,
        asset_reader: &dyn ErasedAssetReader,
        path: &AssetPath<'_>,
    ) -> Result<AssetHash, ReadAssetBytesError> {
        if !self.populate_hashes {
            return Ok(Default::default());
        }
        let meta_bytes = asset_reader.read_meta_bytes(path.path()).await?;
        let minimal: ProcessedInfoMinimal =
            ron::de::from_bytes(&meta_bytes).map_err(DeserializeMetaError::DeserializeMinimal)?;
        let processed_info = minimal
            .processed_info
            .ok_or(ReadAssetBytesError::MissingAssetHash)?;
        Ok(processed_info.full_hash)
    }

    /// Returns a handle to an asset of type `A` with the label `label`. This [`LoadContext`] must produce an asset of the
    /// given type and the given label or the dependencies of this asset will never be considered "fully loaded". However you
    /// can call this method before _or_ after adding the labeled asset.
    pub fn get_label_handle<'b, A: Asset>(
        &mut self,
        label: impl Into<CowArc<'b, str>>,
    ) -> Handle<A> {
        let path = self.asset_path.clone().with_label(label);
        let handle = self.asset_server.get_or_create_path_handle(path, None);
        // `get_or_create_path_handle` always returns a Strong variant, so we are safe to unwrap.
        let index = (&handle).try_into().unwrap();
        self.dependencies.insert(index);
        handle
    }

    /// Returns the labeled asset if it exists.
    pub fn get_labeled(&self, label: impl AsRef<str>) -> Option<&ErasedLoadedAsset> {
        let index = self.label_to_asset_index.get(label.as_ref())?;
        let labeled = &self.labeled_assets[*index];
        Some(&labeled.asset)
    }

    /// Returns the labeled asset given its asset ID if it exists.
    ///
    /// This can be used to get the asset from its handle since `&Handle` implements
    /// [`Into<UntypedAssetId>`].
    pub fn get_labeled_by_id(&self, id: impl Into<UntypedAssetId>) -> Option<&ErasedLoadedAsset> {
        let index = self.asset_id_to_asset_index.get(&id.into())?;
        let labeled = &self.labeled_assets[*index];
        Some(&labeled.asset)
    }

    pub(crate) async fn load_direct_internal(
        &mut self,
        path: AssetPath<'static>,
        settings: &dyn Settings,
        loader: &dyn ErasedAssetLoader,
        reader: &mut dyn Reader,
        processed_info: Option<&ProcessedInfo>,
    ) -> Result<ErasedLoadedAsset, LoadDirectError> {
        let loaded_asset = self
            .asset_server
            .load_with_settings_loader_and_reader(
                &path,
                settings,
                loader,
                reader,
                self.should_load_dependencies,
                self.populate_hashes,
                true,
            )
            .await
            .map_err(|error| LoadDirectError::LoadError {
                dependency: path.clone(),
                error: Box::new(error),
            })?;
        let hash = processed_info.map(|i| i.full_hash).unwrap_or_default();
        self.loader_dependencies.insert(path, hash);
        Ok(loaded_asset)
    }

    /// Create a builder for loading nested assets in this context.
    #[must_use]
    pub fn load_builder(&mut self) -> NestedLoadBuilder<'a, '_> {
        NestedLoadBuilder::new(self)
    }

    /// Retrieves a handle for the asset at the given path and adds that path as a dependency of the asset.
    /// If the current context is a normal [`AssetServer::load`], an actual asset load will be kicked off immediately, which ensures the load happens
    /// as soon as possible.
    /// "Normal loads" kicked from within a normal Bevy App will generally configure the context to kick off loads immediately.
    /// If the current context is configured to not load dependencies automatically (ex: [`AssetProcessor`](crate::processor::AssetProcessor)),
    /// a load will not be kicked off automatically. It is then the calling context's responsibility to begin a load if necessary.
    ///
    /// If you need to override asset settings, asset type, or load directly, please see [`LoadContext::load_builder`].
    pub fn load<'b, A: Asset>(&mut self, path: impl Into<AssetPath<'b>>) -> Handle<A> {
        self.load_builder().load(path)
    }
}

/// An error produced when calling [`LoadContext::read_asset_bytes`]
#[derive(Error, Debug)]
pub enum ReadAssetBytesError {
    #[error("Attempted to load an asset with an empty path \"{0}\"")]
    EmptyPath(AssetPath<'static>),
    #[error(transparent)]
    DeserializeMetaError(#[from] DeserializeMetaError),
    #[error(transparent)]
    AssetReaderError(#[from] AssetReaderError),
    #[error(transparent)]
    MissingAssetSourceError(#[from] MissingAssetSourceError),
    #[error(transparent)]
    MissingProcessedAssetReaderError(#[from] MissingProcessedAssetReaderError),
    /// Encountered an I/O error while loading an asset.
    #[error("Encountered an io error while loading asset at `{}`: {source}", path.display())]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("The LoadContext for this read_asset_bytes call requires hash metadata, but it was not provided. This is likely an internal implementation error.")]
    MissingAssetHash,
}

#[cfg(test)]
mod tests {
    use crate::{
        io::{AssetSourceEvent, Reader},
        tests::{create_app, create_app_with_source_event_sender, run_app_until},
        Asset, AssetApp, AssetEvent, AssetLoadFailedEvent, AssetLoader, AssetServer, Assets,
        DeferredLoadMode, ErasedAssetIndex, Handle, LoadContext, LoadState,
        RecursiveDependencyLoadState,
    };
    use alloc::{boxed::Box, format, string::String, vec, vec::Vec};
    use async_channel::{Receiver, Sender};
    use bevy_app::App;
    use bevy_ecs::{message::Messages, world::World};
    use bevy_reflect::TypePath;
    use futures_lite::AsyncReadExt;
    use std::path::{Path, PathBuf};

    #[derive(Asset, TypePath)]
    struct Root {
        #[dependency]
        deferred: Handle<Deferred>,
        #[dependency]
        other: Handle<Deferred>,
    }

    #[derive(Asset, TypePath, Debug)]
    struct Deferred {
        value: u32,
        #[dependency]
        nested: Option<Handle<Deferred>>,
    }

    /// The labels of the deferred assets of a [`Root`].
    const LABELS: [&str; 2] = ["deferred", "other"];

    /// Reports how a deferred load of the given label went.
    #[derive(Debug, PartialEq, Eq)]
    enum Outcome {
        /// It got through its gate.
        Started(&'static str),
        /// Its work was dropped, after finishing or not.
        Dropped(&'static str),
    }

    struct SendOnDrop(&'static str, crossbeam_channel::Sender<Outcome>);

    impl Drop for SendOnDrop {
        fn drop(&mut self) {
            let _ = self.1.send(Outcome::Dropped(self.0));
        }
    }

    /// Loads a [`Root`] whose labels in [`LABELS`] are deferred. Each waits for a message on its
    /// gate, reads its value from `<label>.txt` and adds a `<label>_nested` labeled asset. The
    /// root also depends on `lazy.txt` without reading it.
    #[derive(TypePath)]
    struct DeferringLoader {
        mode: DeferredLoadMode,
        gates: [Receiver<()>; 2],
        fail: bool,
        outcomes: crossbeam_channel::Sender<Outcome>,
    }

    impl AssetLoader for DeferringLoader {
        type Asset = Root;
        type Settings = ();
        type Error = std::io::Error;

        async fn load(
            &self,
            reader: &mut dyn Reader,
            _settings: &Self::Settings,
            load_context: &mut LoadContext<'_>,
        ) -> Result<Self::Asset, Self::Error> {
            let mut text = String::new();
            reader.read_to_string(&mut text).await?;
            load_context
                .add_loader_dependency("lazy.txt")
                .await
                .map_err(std::io::Error::other)?;

            let [deferred, other] = core::array::from_fn(|index| {
                let label = LABELS[index];
                let gate = self.gates[index].clone();
                let fail = self.fail;
                let outcomes = self.outcomes.clone();
                let on_drop = SendOnDrop(label, self.outcomes.clone());
                load_context.defer_labeled_asset(label, self.mode, move |load_context| {
                    Box::pin(async move {
                        let _on_drop = on_drop;
                        gate.recv().await.map_err(std::io::Error::other)?;
                        outcomes.send(Outcome::Started(label)).unwrap();
                        if fail {
                            return Err(std::io::Error::other("deferred load failed"));
                        }
                        let bytes = load_context
                            .read_asset_bytes(format!("{label}.txt"))
                            .await
                            .map_err(std::io::Error::other)?;
                        let value = String::from_utf8(bytes)
                            .map_err(std::io::Error::other)?
                            .trim()
                            .parse()
                            .map_err(std::io::Error::other)?;
                        let nested = load_context.add_labeled_asset(
                            format!("{label}_nested"),
                            Deferred {
                                value: value + 1,
                                nested: None,
                            },
                        );
                        Ok(Deferred {
                            value,
                            nested: Some(nested),
                        })
                    })
                })
            });
            Ok(Root { deferred, other })
        }

        fn extensions(&self) -> &[&str] {
            &["defer"]
        }
    }

    struct Harness {
        app: App,
        server: AssetServer,
        gates: [Sender<()>; 2],
        outcomes: crossbeam_channel::Receiver<Outcome>,
    }

    impl Harness {
        /// Lets one load of each label through its gate.
        fn open_all(&self) {
            for gate in &self.gates {
                gate.send_blocking(()).unwrap();
            }
        }

        fn outcomes(&self) -> Vec<Outcome> {
            self.outcomes.try_iter().collect()
        }

        fn deferred_value(&self, handle: &Handle<Deferred>) -> Option<u32> {
            self.app
                .world()
                .resource::<Assets<Deferred>>()
                .get(handle)
                .map(|deferred| deferred.value)
        }

        fn started_load_tasks(&self) -> usize {
            self.server.read_infos().stats.started_load_tasks
        }
    }

    fn setup(mode: DeferredLoadMode, fail: bool) -> Harness {
        let (app, dir) = create_app();
        setup_in(app, &dir, mode, fail)
    }

    fn setup_in(
        mut app: App,
        dir: &crate::io::memory::Dir,
        mode: DeferredLoadMode,
        fail: bool,
    ) -> Harness {
        dir.insert_asset_text(Path::new("root.defer"), "");
        dir.insert_asset_text(Path::new("deferred.txt"), "7");
        dir.insert_asset_text(Path::new("other.txt"), "17");
        let (gate, gate_receiver) = async_channel::unbounded();
        let (other_gate, other_gate_receiver) = async_channel::unbounded();
        let (outcome_sender, outcomes) = crossbeam_channel::unbounded();
        app.init_asset::<Root>()
            .init_asset::<Deferred>()
            .register_asset_loader(DeferringLoader {
                mode,
                gates: [gate_receiver, other_gate_receiver],
                fail,
                outcomes: outcome_sender,
            });
        let server = app.world().resource::<AssetServer>().clone();
        Harness {
            app,
            server,
            gates: [gate, other_gate],
            outcomes,
        }
    }

    fn drain_events<A: Asset>(world: &mut World) -> Vec<AssetEvent<A>> {
        world
            .resource_mut::<Messages<AssetEvent<A>>>()
            .drain()
            .collect()
    }

    /// Waits for `root` to be committed and returns the handles of its deferred assets.
    fn wait_for_root(harness: &mut Harness, root: &Handle<Root>) -> [Handle<Deferred>; 2] {
        let server = harness.server.clone();
        run_app_until(&mut harness.app, |_| {
            server.load_state(root).is_loaded().then_some(())
        });
        let root = harness
            .app
            .world()
            .resource::<Assets<Root>>()
            .get(root)
            .unwrap();
        [root.deferred.clone(), root.other.clone()]
    }

    #[test]
    fn eager_deferred_labeled_assets_are_committed_after_root() {
        let mut harness = setup(DeferredLoadMode::Eager, false);
        let root = harness.server.load::<Root>("root.defer");
        let [deferred, other] = wait_for_root(&mut harness, &root);

        // The root is committed while its deferred assets are still waiting on their gates.
        let server = harness.server.clone();
        assert!(server.load_state(&deferred).is_loading());
        assert!(server.load_state(&other).is_loading());
        assert!(server.recursive_dependency_load_state(&root).is_loading());
        assert_eq!(harness.deferred_value(&deferred), None);
        let world = harness.app.world_mut();
        assert_eq!(
            drain_events::<Root>(world),
            [AssetEvent::Added { id: root.id() }]
        );
        assert!(drain_events::<Deferred>(world).is_empty());

        harness.gates[0].send_blocking(()).unwrap();
        run_app_until(&mut harness.app, |_| {
            server.is_loaded_with_dependencies(&deferred).then_some(())
        });
        assert!(server.load_state(&other).is_loading());
        assert!(server.recursive_dependency_load_state(&root).is_loading());

        harness.gates[1].send_blocking(()).unwrap();
        let mut root_events = Vec::new();
        run_app_until(&mut harness.app, |world| {
            root_events.extend(drain_events::<Root>(world));
            (!root_events.is_empty()).then_some(())
        });
        assert_eq!(
            root_events,
            [AssetEvent::LoadedWithDependencies { id: root.id() }]
        );
        assert!(server.recursive_dependency_load_state(&root).is_loaded());

        let deferred_assets = harness.app.world().resource::<Assets<Deferred>>();
        let deferred_value = deferred_assets.get(&deferred).unwrap();
        assert_eq!(deferred_value.value, 7);
        assert_eq!(deferred_assets.get(&other).unwrap().value, 17);
        // Labeled assets added by the deferred load are committed with it.
        let nested = deferred_value.nested.as_ref().unwrap();
        assert_eq!(nested.path(), Some(&"root.defer#deferred_nested".into()));
        assert_eq!(deferred_assets.get(nested).unwrap().value, 8);
    }

    #[test]
    fn deferred_labeled_asset_failure_is_a_dependency_failure() {
        let mut harness = setup(DeferredLoadMode::Eager, true);
        harness.open_all();
        let root = harness.server.load::<Root>("root.defer");
        let [deferred, _other] = wait_for_root(&mut harness, &root);

        let server = harness.server.clone();
        let mut failures = Vec::new();
        run_app_until(&mut harness.app, |world| {
            failures.extend(
                world
                    .resource_mut::<Messages<AssetLoadFailedEvent<Deferred>>>()
                    .drain(),
            );
            (failures.len() == 2).then_some(())
        });
        let failure = failures
            .iter()
            .find(|failure| failure.id == deferred.id())
            .unwrap();
        assert_eq!(failure.path, "root.defer#deferred".into());
        assert!(matches!(server.load_state(&deferred), LoadState::Failed(_)));
        assert!(server.load_state(&root).is_loaded());
        assert!(matches!(
            server.recursive_dependency_load_state(&root),
            RecursiveDependencyLoadState::Failed(_)
        ));
    }

    #[test]
    fn dropping_root_cancels_eager_deferred_labeled_assets() {
        let mut harness = setup(DeferredLoadMode::Eager, false);
        let root = harness.server.load::<Root>("root.defer");
        let deferred_ids = wait_for_root(&mut harness, &root).map(|handle| handle.id());

        // The root holds the only handles to the deferred assets, so dropping it cancels their
        // loads before they get through their gates.
        drop(root);
        let mut outcomes = Vec::new();
        run_app_until(&mut harness.app, |_| {
            outcomes.extend(harness.outcomes.try_iter());
            (outcomes.len() == 2).then_some(())
        });
        assert!(outcomes
            .iter()
            .all(|outcome| matches!(outcome, Outcome::Dropped(_))));
        for id in deferred_ids {
            assert!(harness.server.get_load_state(id).is_none());
        }

        // Opening the gates afterwards does nothing.
        harness.open_all();
        for _ in 0..10 {
            harness.app.update();
        }
        assert!(drain_events::<Deferred>(harness.app.world_mut())
            .iter()
            .all(|event| matches!(event, AssetEvent::Unused { .. })));
        assert_eq!(harness.outcomes(), []);
    }

    #[test]
    fn deferred_labeled_asset_can_be_requested_directly() {
        for mode in [DeferredLoadMode::Eager, DeferredLoadMode::OnDemand] {
            let mut harness = setup(mode, false);
            harness.open_all();
            let deferred = harness.server.load::<Deferred>("root.defer#deferred");
            assert!(harness.server.load_state(&deferred).is_loading());
            let server = harness.server.clone();
            run_app_until(&mut harness.app, |_| {
                server.is_loaded_with_dependencies(&deferred).then_some(())
            });
            assert_eq!(harness.deferred_value(&deferred), Some(7), "{mode:?}");
        }
    }

    #[test]
    fn reload_reproduces_eager_deferred_labeled_assets() {
        let mut harness = setup(DeferredLoadMode::Eager, false);
        harness.open_all();
        let root = harness.server.load::<Root>("root.defer");
        let [deferred, _other] = wait_for_root(&mut harness, &root);
        let server = harness.server.clone();
        run_app_until(&mut harness.app, |_| {
            server.is_loaded_with_dependencies(&root).then_some(())
        });
        drain_events::<Root>(harness.app.world_mut());
        drain_events::<Deferred>(harness.app.world_mut());

        // While the reloaded deferred assets wait on their gates, the root is committed and waits
        // for them, and the previous values stay available.
        server.reload("root.defer");
        let mut root_events = Vec::new();
        run_app_until(&mut harness.app, |world| {
            root_events.extend(drain_events::<Root>(world));
            (!root_events.is_empty()).then_some(())
        });
        assert_eq!(root_events, [AssetEvent::Modified { id: root.id() }]);
        assert!(server.load_state(&deferred).is_loading());
        assert!(server.recursive_dependency_load_state(&root).is_loading());
        assert_eq!(harness.deferred_value(&deferred), Some(7));

        harness.open_all();
        let mut deferred_events = Vec::new();
        run_app_until(&mut harness.app, |world| {
            deferred_events.extend(drain_events::<Deferred>(world));
            server.is_loaded_with_dependencies(&root).then_some(())
        });
        assert!(deferred_events.contains(&AssetEvent::Modified { id: deferred.id() }));
    }

    #[test]
    fn on_demand_labeled_assets_wait_to_be_requested() {
        let mut harness = setup(DeferredLoadMode::OnDemand, false);
        harness.open_all();
        let root = harness.server.load::<Root>("root.defer");
        let [deferred, other] = wait_for_root(&mut harness, &root);

        // The root does not wait for on-demand assets, and holding their handles does not start
        // them.
        let server = harness.server.clone();
        run_app_until(&mut harness.app, |_| {
            server.is_loaded_with_dependencies(&root).then_some(())
        });
        for _ in 0..10 {
            harness.app.update();
        }
        for handle in [&deferred, &other] {
            assert!(matches!(server.load_state(handle), LoadState::NotLoaded));
            assert_eq!(harness.deferred_value(handle), None);
        }
        assert_eq!(harness.outcomes(), []);

        // Requesting one starts only that one, without loading the root again.
        let requested = server.load::<Deferred>("root.defer#deferred");
        assert_eq!(requested, deferred);
        assert!(server.load_state(&requested).is_loading());
        run_app_until(&mut harness.app, |_| {
            server.is_loaded_with_dependencies(&requested).then_some(())
        });
        assert_eq!(harness.deferred_value(&deferred), Some(7));
        let nested = harness
            .server
            .get_handle::<Deferred>("root.defer#deferred_nested");
        assert_eq!(harness.deferred_value(&nested.unwrap()), Some(8));
        assert_eq!(
            harness.outcomes(),
            [Outcome::Started("deferred"), Outcome::Dropped("deferred")]
        );
        assert!(matches!(server.load_state(&other), LoadState::NotLoaded));
        assert!(server.is_loaded_with_dependencies(&root));
        assert_eq!(harness.started_load_tasks(), 1);
    }

    #[test]
    fn on_demand_labeled_asset_failure_does_not_fail_root() {
        let mut harness = setup(DeferredLoadMode::OnDemand, true);
        harness.open_all();
        let root = harness.server.load::<Root>("root.defer");
        wait_for_root(&mut harness, &root);
        let server = harness.server.clone();
        let deferred = server.load::<Deferred>("root.defer#deferred");
        run_app_until(&mut harness.app, |_| {
            server.load_state(&deferred).is_failed().then_some(())
        });
        assert!(server.is_loaded_with_dependencies(&root));
    }

    #[test]
    fn dropping_requested_on_demand_labeled_asset_cancels_it() {
        let mut harness = setup(DeferredLoadMode::OnDemand, false);
        let root = harness.server.load::<Root>("root.defer");
        let held = wait_for_root(&mut harness, &root);
        let server = harness.server.clone();
        let requested = server.load::<Deferred>("root.defer#deferred");
        let index: ErasedAssetIndex = (&requested).try_into().unwrap();
        assert!(server.load_state(&requested).is_loading());
        assert!(server.read_infos().deferred_tasks.contains_key(&index));

        // Dropping every handle, including the ones the root holds, cancels the started load and
        // drops the work of the one that was never requested.
        drop((root, held, requested));
        let mut outcomes = Vec::new();
        run_app_until(&mut harness.app, |_| {
            outcomes.extend(harness.outcomes.try_iter());
            (outcomes.len() == 2).then_some(())
        });
        outcomes.sort_by_key(|outcome| format!("{outcome:?}"));
        assert_eq!(
            outcomes,
            [Outcome::Dropped("deferred"), Outcome::Dropped("other")]
        );
        assert!(!server.read_infos().deferred_tasks.contains_key(&index));

        harness.open_all();
        for _ in 0..10 {
            harness.app.update();
        }
        assert_eq!(harness.outcomes(), []);
    }

    #[test]
    fn reload_restarts_requested_on_demand_labeled_assets() {
        let mut harness = setup(DeferredLoadMode::OnDemand, false);
        harness.open_all();
        harness.open_all();
        let root = harness.server.load::<Root>("root.defer");
        let [deferred, other] = wait_for_root(&mut harness, &root);
        let server = harness.server.clone();
        let _requested = server.load::<Deferred>("root.defer#deferred");
        run_app_until(&mut harness.app, |_| {
            server.is_loaded_with_dependencies(&deferred).then_some(())
        });
        harness.outcomes();
        drain_events::<Deferred>(harness.app.world_mut());

        server.reload("root.defer");
        let mut deferred_events = Vec::new();
        run_app_until(&mut harness.app, |world| {
            deferred_events.extend(drain_events::<Deferred>(world));
            deferred_events
                .contains(&AssetEvent::Modified { id: deferred.id() })
                .then_some(())
        });
        // The work of `other` from the first load is replaced without starting.
        let outcomes = harness.outcomes();
        assert!(outcomes.contains(&Outcome::Started("deferred")));
        assert!(!outcomes.contains(&Outcome::Started("other")));
        assert!(matches!(server.load_state(&other), LoadState::NotLoaded));
        assert!(server.is_loaded_with_dependencies(&root));
    }

    #[test]
    fn deferred_labeled_asset_reads_reload_the_root() {
        let (app, dir, source_events) = create_app_with_source_event_sender();
        let mut harness = setup_in(app, &dir, DeferredLoadMode::Eager, false);
        for _ in 0..3 {
            harness.open_all();
        }
        let root = harness.server.load::<Root>("root.defer");
        let [deferred, _other] = wait_for_root(&mut harness, &root);
        let server = harness.server.clone();
        run_app_until(&mut harness.app, |_| {
            server.is_loaded_with_dependencies(&root).then_some(())
        });
        let root_path = crate::AssetPath::from("root.defer");
        for dependency in ["deferred.txt", "lazy.txt"] {
            let dependents =
                server.read_infos().loader_dependents[&crate::AssetPath::from(dependency)].clone();
            assert_eq!(dependents.len(), 1);
            assert!(dependents.contains(&root_path));
        }

        // Changing a file a deferred asset read reloads the root once, which reproduces it.
        dir.insert_asset_text(Path::new("deferred.txt"), "8");
        source_events
            .send_blocking(AssetSourceEvent::ModifiedAsset(PathBuf::from(
                "deferred.txt",
            )))
            .unwrap();
        run_app_until(&mut harness.app, |world| {
            let value = world
                .resource::<Assets<Deferred>>()
                .get(&deferred)
                .map(|d| d.value);
            (value == Some(8)).then_some(())
        });
        assert_eq!(harness.started_load_tasks(), 2);

        // So does changing a file the root depends on without reading it.
        source_events
            .send_blocking(AssetSourceEvent::ModifiedAsset(PathBuf::from("lazy.txt")))
            .unwrap();
        run_app_until(&mut harness.app, |_| {
            (server.read_infos().stats.started_load_tasks == 3).then_some(())
        });
    }

    #[derive(Asset, TypePath)]
    struct Embedding(Vec<u32>);

    /// Loads a [`Root`] as a value, and records the deferred asset's value.
    #[derive(TypePath)]
    struct EmbeddingLoader;

    impl AssetLoader for EmbeddingLoader {
        type Asset = Embedding;
        type Settings = ();
        type Error = crate::LoadDirectError;

        async fn load(
            &self,
            _reader: &mut dyn Reader,
            _settings: &Self::Settings,
            load_context: &mut LoadContext<'_>,
        ) -> Result<Self::Asset, Self::Error> {
            let root = load_context
                .load_builder()
                .load_value::<Root>("root.defer")
                .await?;
            // The deferred asset is a labeled asset of the result like any other, holding its own
            // labeled assets.
            let deferred = root.get_labeled("deferred").unwrap();
            let nested = deferred.get_labeled("deferred_nested").unwrap();
            let values = [deferred, nested]
                .into_iter()
                .map(|labeled| labeled.get::<Deferred>().unwrap().value)
                .collect();
            Ok(Embedding(values))
        }

        fn extensions(&self) -> &[&str] {
            &["embed"]
        }
    }

    #[test]
    fn immediate_load_runs_deferred_labeled_assets() {
        for mode in [DeferredLoadMode::Eager, DeferredLoadMode::OnDemand] {
            let (app, dir) = create_app();
            dir.insert_asset_text(Path::new("a.embed"), "");
            let mut harness = setup_in(app, &dir, mode, false);
            harness
                .app
                .init_asset::<Embedding>()
                .register_asset_loader(EmbeddingLoader);
            harness.open_all();

            let embedding = harness.server.load("a.embed");
            run_app_until(&mut harness.app, |world| {
                world
                    .resource::<Assets<Embedding>>()
                    .get(&embedding)
                    .map(|embedding| assert_eq!(embedding.0, vec![7, 8], "{mode:?}"))
            });
        }
    }
}
