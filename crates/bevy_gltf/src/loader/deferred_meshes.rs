//! Building primitive meshes as deferred labeled assets, for
//! [`GltfLoaderSettings::defer_meshes`](super::GltfLoaderSettings::defer_meshes).
//!
//! The load itself only reads the buffers that the file holds inline or that something other
//! than a mesh uses: images, animations, skins and instance transforms. Each primitive reads the
//! other buffers it needs when it is built. A buffer read for one primitive stays cached until
//! the last primitive that uses it is built or cancelled.

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use bevy_asset::{AssetPath, DeferredLoadMode, Handle, LoadContext};
use bevy_mesh::Mesh;
use bevy_platform::sync::OnceLock;
use core::sync::atomic::{AtomicUsize, Ordering};
use fixedbitset::FixedBitSet;
use gltf::buffer::Source;

use super::{finish_primitive_mesh, read_primitive_mesh, DataUri, GltfError, PrimitiveMeshOptions};
use crate::GltfAssetLabel;

/// The buffers the load of `gltf` reads itself when its meshes are deferred.
pub(super) fn eager_buffers(gltf: &gltf::Gltf) -> FixedBitSet {
    let mut eager = FixedBitSet::with_capacity(gltf.buffers().len());
    for buffer in gltf.buffers() {
        // Fallback buffers have no URI either, but hold no data until decoded.
        #[cfg(feature = "meshopt")]
        if super::extensions::khr_meshopt_compression::is_fallback_buffer(&buffer) {
            continue;
        }
        // These are already in memory with the glTF itself.
        let inline = match buffer.source() {
            Source::Bin => true,
            Source::Uri(uri) => percent_encoding::percent_decode_str(uri)
                .decode_utf8()
                .is_ok_and(|uri| DataUri::parse(&uri).is_ok()),
        };
        eager.set(buffer.index(), inline);
    }

    let mut views = Vec::new();
    for image in gltf.images() {
        if let gltf::image::Source::View { view, .. } = image.source() {
            views.push(view);
        }
    }
    let mut accessors = Vec::new();
    for animation in gltf.animations() {
        for sampler in animation.samplers() {
            accessors.extend([sampler.input(), sampler.output()]);
        }
    }
    accessors.extend(gltf.skins().filter_map(|skin| skin.inverse_bind_matrices()));
    for node in gltf.nodes() {
        accessors.extend(super::extensions::instance_transform_accessors(
            &gltf.document,
            &node,
        ));
    }
    for accessor in &accessors {
        views.extend(accessor_views(accessor));
    }

    for view in views {
        eager.insert(view.buffer().index());
    }
    // The load decodes every view of the fallback buffers it reads, so it needs their sources.
    #[cfg(feature = "meshopt")]
    for view in gltf.views() {
        if eager.contains(view.buffer().index())
            && let Ok(Some(source)) =
                super::extensions::khr_meshopt_compression::compressed_source(&view)
            && source < eager.len()
        {
            eager.insert(source);
        }
    }
    eager
}

/// The buffer views `accessor` reads.
fn accessor_views<'a>(
    accessor: &gltf::Accessor<'a>,
) -> impl Iterator<Item = gltf::buffer::View<'a>> {
    let sparse = accessor.sparse();
    accessor.view().into_iter().chain(
        sparse
            .into_iter()
            .flat_map(|sparse| [sparse.indices().view(), sparse.values().view()]),
    )
}

/// Where a buffer that the load did not read comes from.
enum LazySource {
    /// A file next to the glTF.
    Uri(AssetPath<'static>),
    /// A meshopt fallback buffer, decoded from the buffers its views compress into.
    #[cfg(feature = "meshopt")]
    Fallback { length: usize },
}

struct Buffer {
    /// `None` for buffers the load read itself.
    source: Option<LazySource>,
    /// The number of deferred primitives yet to be built that read this buffer.
    users: AtomicUsize,
    data: async_lock::Mutex<Option<Arc<Vec<u8>>>>,
}

/// State shared by the deferred primitive meshes of one glTF load.
pub(super) struct DeferredMeshes {
    document: gltf::Document,
    mode: DeferredLoadMode,
    options: PrimitiveMeshOptions,
    buffers: Vec<Buffer>,
    /// The buffers the load read, set when it finishes. Lazy buffers are empty here.
    eager: OnceLock<Vec<Vec<u8>>>,
}

/// Counts a deferred primitive as a user of a lazy buffer until it is built or cancelled.
struct BufferUse {
    meshes: Arc<DeferredMeshes>,
    buffer: usize,
}

impl BufferUse {
    fn new(meshes: &Arc<DeferredMeshes>, buffer: usize) -> Self {
        meshes.buffers[buffer].users.fetch_add(1, Ordering::AcqRel);
        Self {
            meshes: meshes.clone(),
            buffer,
        }
    }
}

impl Drop for BufferUse {
    fn drop(&mut self) {
        let buffer = &self.meshes.buffers[self.buffer];
        if buffer.users.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(mut data) = buffer.data.try_lock()
        {
            // The last user is done, so release the data. If another task holds the lock, it is
            // reading this as a meshopt source and does not cache it.
            *data = None;
        }
    }
}

impl DeferredMeshes {
    pub(super) fn new(
        gltf: &gltf::Gltf,
        path: &AssetPath<'static>,
        eager: &FixedBitSet,
        mode: DeferredLoadMode,
        options: PrimitiveMeshOptions,
    ) -> Result<Arc<Self>, GltfError> {
        let buffers = gltf
            .buffers()
            .map(|buffer| {
                let source = if eager.contains(buffer.index()) {
                    None
                } else {
                    Some(lazy_source(&buffer, path)?)
                };
                Ok(Buffer {
                    source,
                    users: AtomicUsize::new(0),
                    data: async_lock::Mutex::new(None),
                })
            })
            .collect::<Result<_, GltfError>>()?;
        Ok(Arc::new(Self {
            document: gltf.document.clone(),
            mode,
            options,
            buffers,
            eager: OnceLock::new(),
        }))
    }

    /// Makes the load depend on the buffer files it leaves to the deferred meshes, so changing
    /// one reloads the glTF even if no mesh has read it yet.
    pub(super) async fn add_loader_dependencies(
        &self,
        load_context: &mut LoadContext<'_>,
    ) -> Result<(), GltfError> {
        for buffer in &self.buffers {
            if let Some(LazySource::Uri(path)) = &buffer.source {
                load_context.add_loader_dependency(path.clone()).await?;
            }
        }
        Ok(())
    }

    /// Provides the buffers the load read. Must be called before the load returns.
    pub(super) fn set_eager_buffers(&self, buffer_data: Vec<Vec<u8>>) {
        let _ = self.eager.set(buffer_data);
    }

    /// Defers building the mesh of `primitive` of `gltf_mesh`, returning its handle.
    pub(super) fn defer_primitive(
        self: &Arc<Self>,
        load_context: &mut LoadContext,
        gltf_mesh: &gltf::Mesh,
        primitive: &gltf::Primitive,
        label: GltfAssetLabel,
        on_skinned_nodes: bool,
        on_non_skinned_nodes: bool,
    ) -> Handle<Mesh> {
        let mut lazy = FixedBitSet::with_capacity(self.buffers.len());
        let accessors = primitive
            .attributes()
            .map(|(_, accessor)| accessor)
            .chain(primitive.indices())
            .chain(primitive.morph_targets().flat_map(|target| {
                [target.positions(), target.normals(), target.tangents()]
                    .into_iter()
                    .flatten()
            }));
        for accessor in accessors {
            for view in accessor_views(&accessor) {
                let buffer = view.buffer().index();
                if self.buffers.get(buffer).is_some_and(|b| b.source.is_some()) {
                    lazy.insert(buffer);
                }
            }
        }
        let uses: Vec<_> = lazy
            .ones()
            .map(|buffer| BufferUse::new(self, buffer))
            .collect();

        if self.mode == DeferredLoadMode::Eager {
            // `Gltf` does not visit the handles it holds, so make the load depend on the mesh
            // explicitly. The handle is the one `defer_labeled_asset` returns.
            load_context.get_label_handle::<Mesh>(label.to_string());
        }

        let meshes = self.clone();
        let mesh_index = gltf_mesh.index();
        let primitive_index = primitive.index();
        load_context.defer_labeled_asset(label.to_string(), self.mode, move |load_context| {
            Box::pin(async move {
                meshes
                    .build_primitive(
                        load_context,
                        mesh_index,
                        primitive_index,
                        label,
                        uses,
                        on_skinned_nodes,
                        on_non_skinned_nodes,
                    )
                    .await
            })
        })
    }

    async fn build_primitive(
        &self,
        load_context: &mut LoadContext<'_>,
        mesh_index: usize,
        primitive_index: usize,
        label: GltfAssetLabel,
        uses: Vec<BufferUse>,
        on_skinned_nodes: bool,
        on_non_skinned_nodes: bool,
    ) -> Result<Mesh, GltfError> {
        let mut lazy = Vec::with_capacity(uses.len());
        for buffer_use in &uses {
            lazy.push((
                buffer_use.buffer,
                self.buffer(buffer_use.buffer, load_context).await?,
            ));
        }
        let mut buffer_data: Vec<&[u8]> = self.eager().iter().map(Vec::as_slice).collect();
        for (buffer, data) in &lazy {
            buffer_data[*buffer] = data;
        }

        // The document is the one this primitive came from, so these exist.
        let gltf_mesh = self.document.meshes().nth(mesh_index).unwrap();
        let primitive = gltf_mesh.primitives().nth(primitive_index).unwrap();
        let mesh = read_primitive_mesh(
            &gltf_mesh,
            &primitive,
            label,
            &buffer_data,
            &self.options,
            on_skinned_nodes,
            on_non_skinned_nodes,
        )?;
        Ok(finish_primitive_mesh(
            mesh,
            &primitive,
            &self.options,
            on_skinned_nodes,
        ))
    }

    fn eager(&self) -> &[Vec<u8>] {
        self.eager
            .get()
            .expect("deferred meshes are built after the load returns")
    }

    /// Reads or decodes lazy buffer `index`, or returns it from the cache.
    async fn buffer(
        &self,
        index: usize,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Arc<Vec<u8>>, GltfError> {
        match &self.buffers[index].source {
            Some(LazySource::Uri(_)) | None => self.uri_buffer(index, load_context).await,
            #[cfg(feature = "meshopt")]
            Some(LazySource::Fallback { length }) => {
                self.cached(index, self.decode_fallback(index, *length, load_context))
                    .await
            }
        }
    }

    /// Reads lazy buffer `index` from its file, or returns it from the cache.
    async fn uri_buffer(
        &self,
        index: usize,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Arc<Vec<u8>>, GltfError> {
        let Some(LazySource::Uri(path)) = &self.buffers[index].source else {
            unreachable!("only lazy buffers with a URI are read from files");
        };
        self.cached(index, async {
            Ok(load_context.read_asset_bytes(path.clone()).await?)
        })
        .await
    }

    /// Returns the cached data of buffer `index`, or loads it with `load`. The result is cached
    /// while primitives that use the buffer remain.
    async fn cached(
        &self,
        index: usize,
        load: impl Future<Output = Result<Vec<u8>, GltfError>>,
    ) -> Result<Arc<Vec<u8>>, GltfError> {
        let buffer = &self.buffers[index];
        let mut data = buffer.data.lock().await;
        if let Some(data) = &*data {
            return Ok(data.clone());
        }
        let loaded = Arc::new(load.await?);
        if buffer.users.load(Ordering::Acquire) > 0 {
            *data = Some(loaded.clone());
        }
        Ok(loaded)
    }

    /// Decodes every compressed view of fallback buffer `index`.
    #[cfg(feature = "meshopt")]
    async fn decode_fallback(
        &self,
        index: usize,
        length: usize,
        load_context: &mut LoadContext<'_>,
    ) -> Result<Vec<u8>, GltfError> {
        use super::extensions::khr_meshopt_compression::{compressed_source, decode_view};

        let mut fallback = Vec::new();
        fallback
            .try_reserve_exact(length)
            .map_err(|_| std::io::Error::from(std::io::ErrorKind::OutOfMemory))?;
        fallback.resize(length, 0);
        for view in self.document.views() {
            if view.buffer().index() != index {
                continue;
            }
            let Some(source) = compressed_source(&view)? else {
                continue;
            };
            let invalid = |message: &str| GltfError::InvalidMeshoptCompression {
                buffer_view: view.index(),
                message: alloc::format!("compressed buffer {source} {message}"),
            };
            let lazy_source;
            let src = match self.buffers.get(source).map(|buffer| &buffer.source) {
                None => return Err(invalid("does not exist")),
                Some(None) => self.eager()[source].as_slice(),
                Some(Some(LazySource::Uri(_))) => {
                    lazy_source = self.uri_buffer(source, load_context).await?;
                    lazy_source.as_slice()
                }
                Some(Some(LazySource::Fallback { .. })) => {
                    return Err(invalid("is itself a fallback buffer"))
                }
            };
            decode_view(&view, src, &mut fallback)?;
        }
        Ok(fallback)
    }
}

/// Where buffer `buffer` of the glTF at `path` comes from, if the load does not read it.
fn lazy_source(buffer: &gltf::Buffer, path: &AssetPath<'static>) -> Result<LazySource, GltfError> {
    #[cfg(feature = "meshopt")]
    if super::extensions::khr_meshopt_compression::is_fallback_buffer(buffer) {
        return Ok(LazySource::Fallback {
            length: buffer.length(),
        });
    }
    match buffer.source() {
        Source::Uri(uri) => {
            let uri = percent_encoding::percent_decode_str(uri)
                .decode_utf8()
                .unwrap();
            let buffer_path = path
                .resolve_embed_str(&uri)
                .map_err(|err| GltfError::InvalidBufferUri(uri.as_ref().into(), err))?;
            Ok(LazySource::Uri(buffer_path))
        }
        Source::Bin => unreachable!("the binary chunk is always read with the glTF"),
    }
}
