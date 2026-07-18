use crate::{
    archetype::Archetype,
    change_detection::Tick,
    component::{Component, ComponentId, Components, StorageType},
    entity::{Entities, Entity},
    query::{DebugCheckedUnwrap, FilteredAccess, FilteredAccessSet, StorageSwitch, WorldQuery},
    storage::{ComponentSparseSet, Table, TableRow},
    world::{unsafe_world_cell::UnsafeWorldCell, World},
};
use bevy_ptr::{ThinSlicePtr, UnsafeCellDeref};
use bevy_utils::prelude::DebugName;
use core::{cell::UnsafeCell, marker::PhantomData};
use nonmax::NonMaxU32;
use variadics_please::all_tuples;

/// Types that filter the results of a [`Query`].
///
/// There are many types that natively implement this trait:
/// - **Component filters.**
///   [`With`] and [`Without`] filters can be applied to check if the queried entity does or does not contain a particular component.
/// - **Change detection filters.**
///   [`Added`] and [`Changed`] filters can be applied to detect component changes to an entity.
/// - **Spawned filter.**
///   [`Spawned`] filter can be applied to check if the queried entity was spawned recently.
/// - **`QueryFilter` tuples.**
///   If every element of a tuple implements `QueryFilter`, then the tuple itself also implements the same trait.
///   This enables a single `Query` to filter over multiple conditions.
///   Due to the current lack of variadic generics in Rust, the trait has been implemented for tuples from 0 to 15 elements,
///   but nesting of tuples allows infinite `QueryFilter`s.
/// - **Filter disjunction operator.**
///   By default, tuples compose query filters in such a way that all conditions must be satisfied to generate a query item for a given entity.
///   Wrapping a tuple inside an [`Or`] operator will relax the requirement to just one condition.
///
/// Implementing the trait manually can allow for a fundamentally new type of behavior.
///
/// Query design can be easily structured by deriving `QueryFilter` for custom types.
/// Despite the added complexity, this approach has several advantages over using `QueryFilter` tuples.
/// The most relevant improvements are:
///
/// - Reusability across multiple systems.
/// - Filters can be composed together to create a more complex filter.
///
/// This trait can only be derived for structs if each field also implements `QueryFilter`.
///
/// ```
/// # use bevy_ecs::prelude::*;
/// # use bevy_ecs::{query::QueryFilter, component::Component};
/// #
/// # #[derive(Component)]
/// # struct ComponentA;
/// # #[derive(Component)]
/// # struct ComponentB;
/// # #[derive(Component)]
/// # struct ComponentC;
/// # #[derive(Component)]
/// # struct ComponentD;
/// # #[derive(Component)]
/// # struct ComponentE;
/// #
/// #[derive(QueryFilter)]
/// struct MyFilter<T: Component, P: Component> {
///     // Field names are not relevant, since they are never manually accessed.
///     with_a: With<ComponentA>,
///     or_filter: Or<(With<ComponentC>, Added<ComponentB>)>,
///     generic_tuple: (With<T>, Without<P>),
/// }
///
/// fn my_system(query: Query<Entity, MyFilter<ComponentD, ComponentE>>) {
///     // ...
/// }
/// # bevy_ecs::system::assert_is_system(my_system);
/// ```
///
/// [`Query`]: crate::system::Query
///
/// # Safety
///
/// The [`WorldQuery`] implementation must not take any mutable access.
/// This is the same safety requirement as [`ReadOnlyQueryData`](crate::query::ReadOnlyQueryData).
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid `Query` filter",
    label = "invalid `Query` filter",
    note = "a `QueryFilter` typically uses a combination of `With<T>` and `Without<T>` statements"
)]
pub unsafe trait QueryFilter: WorldQuery {
    /// Returns true if (and only if) this Filter relies strictly on archetypes to limit which
    /// components are accessed by the Query.
    ///
    /// This enables optimizations for [`QueryIter`](`crate::query::QueryIter`) that rely on knowing exactly how
    /// many elements are being iterated (such as `Iterator::collect()`).
    ///
    /// If this is `true`, then [`QueryFilter::filter_fetch`] must always return true.
    const IS_ARCHETYPAL: bool;

    /// Returns true if the provided [`Entity`] and [`TableRow`] should be included in the query results.
    /// If false, the entity will be skipped.
    ///
    /// Note that this is called after already restricting the matched [`Table`]s and [`Archetype`]s to the
    /// ones that are compatible with the Filter's access.
    ///
    /// Implementors of this method will generally either have a trivial `true` body (required for archetypal filters),
    /// or access the necessary data within this function to make the final decision on filter inclusion.
    ///
    /// # Safety
    ///
    /// Must always be called _after_ [`WorldQuery::set_table`] or [`WorldQuery::set_archetype`]. `entity` and
    /// `table_row` must be in the range of the current table and archetype.
    unsafe fn filter_fetch(
        state: &Self::State,
        fetch: &mut Self::Fetch<'_>,
        entity: Entity,
        table_row: TableRow,
    ) -> bool;

    /// Whether [`filter_fetch`](QueryFilter::filter_fetch) is a pure function of the current
    /// table/archetype state: no side effects and no fetch-state mutation, so it may be
    /// evaluated for any in-range row, in any order — including rows that are never yielded,
    /// and including skipping evaluation for rows another filter already rejected.
    ///
    /// When `true`, query iteration may evaluate the filter for a whole block of rows at once
    /// via [`filter_fetch_mask`](QueryFilter::filter_fetch_mask) instead of interleaving one
    /// `filter_fetch` call per candidate row. `false` (the default) preserves the exact
    /// per-row call pattern.
    #[doc(hidden)]
    const BATCHABLE: bool = false;

    /// Evaluates this filter for a contiguous block of up to [`FILTER_MASK_BITS`] table rows.
    ///
    /// Bit `i` of the result reports [`filter_fetch`](QueryFilter::filter_fetch) for the
    /// entity `entities[i]` at table row `first_row + i`. Bits at positions
    /// `>= entities.len()` are unspecified.
    ///
    /// # Safety
    ///
    /// - Must always be called _after_ [`WorldQuery::set_table`] or
    ///   [`WorldQuery::set_archetype`], with [`filter_fetch`](QueryFilter::filter_fetch)'s
    ///   preconditions holding for every row in the block.
    /// - `entities.len()` must be at most [`FILTER_MASK_BITS`], and `entities[i]` must be
    ///   the entity stored at table row `first_row + i` of the current table.
    #[doc(hidden)]
    #[inline]
    unsafe fn filter_fetch_mask(
        state: &Self::State,
        fetch: &mut Self::Fetch<'_>,
        entities: &[Entity],
        first_row: u32,
    ) -> u64
    where
        Self: Sized,
    {
        if Self::IS_ARCHETYPAL {
            // Archetypal filters must always pass `filter_fetch` (see `IS_ARCHETYPAL`).
            return u64::MAX;
        }
        // SAFETY: The caller's guarantees are exactly this helper's requirements.
        unsafe { filter_fetch_mask_fallback::<Self>(state, fetch, entities, first_row) }
    }
}

/// The number of table rows evaluated per [`QueryFilter::filter_fetch_mask`] block — the
/// bit width of the returned mask.
pub(crate) const FILTER_MASK_BITS: u32 = u64::BITS;

/// Scalar fallback for [`QueryFilter::filter_fetch_mask`]: one
/// [`filter_fetch`](QueryFilter::filter_fetch) call per row, in ascending row order.
///
/// Public only for the `QueryFilter` derive macro's generated code.
///
/// # Safety
///
/// Same requirements as [`QueryFilter::filter_fetch_mask`].
#[doc(hidden)]
#[inline]
pub unsafe fn filter_fetch_mask_fallback<F: QueryFilter>(
    state: &F::State,
    fetch: &mut F::Fetch<'_>,
    entities: &[Entity],
    first_row: u32,
) -> u64 {
    let mut mask = 0u64;
    for (i, &entity) in entities.iter().enumerate() {
        // SAFETY: The caller guarantees `first_row + i` is a valid row in the current table,
        // and valid table rows are always less than `u32::MAX`.
        let row = unsafe { TableRow::new(NonMaxU32::new_unchecked(first_row + i as u32)) };
        // SAFETY: The invariants are upheld by the caller.
        let matched = unsafe { F::filter_fetch(state, fetch, entity, row) };
        mask |= (matched as u64) << i;
    }
    mask
}

/// Packs [`FILTER_MASK_BITS`] bytes that are each `0` or `1` into a bitmask with
/// bit `i` = `bools[i]`.
///
/// Split from the compare loops that fill `bools` so those loops stay trivially
/// auto-vectorizable.
#[inline(always)]
fn pack_bools_to_mask(bools: &[u8; FILTER_MASK_BITS as usize]) -> u64 {
    let mut mask = 0u64;
    for j in 0..8 {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&bools[j * 8..j * 8 + 8]);
        let word = u64::from_le_bytes(bytes);
        // Each byte is 0 or 1, so this multiply gathers every byte's low bit into the top
        // byte without cross-term carries.
        mask |= ((word.wrapping_mul(0x0102_0408_1020_4080) >> 56) & 0xFF) << (j * 8);
    }
    mask
}

/// Shared [`QueryFilter::filter_fetch_mask`] implementation for the tick-based filters
/// ([`Added`], [`Changed`]): builds one block's match mask from the cached tick column, or
/// via `sparse_tick` for sparse-set components.
///
/// # Safety
///
/// Same requirements as [`QueryFilter::filter_fetch_mask`], and `sparse_tick` must be
/// sound to call for every entity in `entities`.
#[inline]
unsafe fn tick_filter_fetch_mask<'w, T: Component>(
    ticks: &StorageSwitch<
        T,
        Option<ThinSlicePtr<'w, UnsafeCell<Tick>>>,
        Option<&'w ComponentSparseSet>,
    >,
    this_run: Tick,
    threshold: u32,
    entities: &[Entity],
    first_row: u32,
    sparse_tick: impl Fn(&ComponentSparseSet, Entity) -> Tick,
) -> u64 {
    ticks.extract(
        |table| {
            // SAFETY: set_table was previously called
            let table = unsafe { table.debug_checked_unwrap() };
            let len = entities.len();
            let mut bools = [0u8; FILTER_MASK_BITS as usize];
            if len == FILTER_MASK_BITS as usize {
                // Fixed trip count so the compare loop auto-vectorizes.
                for (i, matched) in bools.iter_mut().enumerate() {
                    // SAFETY: The caller ensures rows `first_row..first_row + FILTER_MASK_BITS`
                    // are in range, and that no exclusive access aliases the tick array.
                    let tick = unsafe { table.get_unchecked(first_row as usize + i).read() };
                    *matched = tick.is_newer_than_threshold(this_run, threshold) as u8;
                }
            } else {
                for (i, matched) in bools[..len].iter_mut().enumerate() {
                    // SAFETY: As above, for rows `first_row..first_row + len`.
                    let tick = unsafe { table.get_unchecked(first_row as usize + i).read() };
                    *matched = tick.is_newer_than_threshold(this_run, threshold) as u8;
                }
            }
            pack_bools_to_mask(&bools)
        },
        |sparse_set| {
            // SAFETY: init_fetch was previously called for a sparse-set component.
            let sparse_set = unsafe { sparse_set.debug_checked_unwrap() };
            let mut mask = 0u64;
            for (i, &entity) in entities.iter().enumerate() {
                let tick = sparse_tick(sparse_set, entity);
                mask |= (tick.is_newer_than_threshold(this_run, threshold) as u64) << i;
            }
            mask
        },
    )
}

/// Filter that selects entities with a component `T`.
///
/// This can be used in a [`Query`](crate::system::Query) if entities are required to have the
/// component `T` but you don't actually care about components value.
///
/// This is the negation of [`Without`].
///
/// # Examples
///
/// ```
/// # use bevy_ecs::component::Component;
/// # use bevy_ecs::query::With;
/// # use bevy_ecs::system::IntoSystem;
/// # use bevy_ecs::system::Query;
/// #
/// # #[derive(Component)]
/// # struct IsBeautiful;
/// # #[derive(Component)]
/// # struct Name { name: &'static str };
/// #
/// fn compliment_entity_system(query: Query<&Name, With<IsBeautiful>>) {
///     for name in &query {
///         println!("{} is looking lovely today!", name.name);
///     }
/// }
/// # bevy_ecs::system::assert_is_system(compliment_entity_system);
/// ```
pub struct With<T>(PhantomData<T>);

// SAFETY:
// `update_component_access` does not add any accesses.
// This is sound because [`QueryFilter::filter_fetch`] does not access any components.
// `update_component_access` adds a `With` filter for `T`.
// This is sound because `matches_component_set` returns whether the set contains the component.
unsafe impl<T: Component> WorldQuery for With<T> {
    type Fetch<'w> = ();
    type State = ComponentId;

    fn shrink_fetch<'wlong: 'wshort, 'wshort>(_: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {}

    #[inline]
    unsafe fn init_fetch(
        _world: UnsafeWorldCell,
        _state: &ComponentId,
        _last_run: Tick,
        _this_run: Tick,
    ) {
    }

    const IS_DENSE: bool = {
        match T::STORAGE_TYPE {
            StorageType::Table => true,
            StorageType::SparseSet => false,
        }
    };

    #[inline]
    unsafe fn set_archetype(
        _fetch: &mut (),
        _state: &ComponentId,
        _archetype: &Archetype,
        _table: &Table,
    ) {
    }

    #[inline]
    unsafe fn set_table(_fetch: &mut (), _state: &ComponentId, _table: &Table) {}

    #[inline]
    fn update_component_access(&id: &ComponentId, access: &mut FilteredAccess) {
        access.and_with(id);
    }

    fn init_state(world: &mut World) -> ComponentId {
        world.register_component::<T>()
    }

    fn get_state(components: &Components) -> Option<Self::State> {
        components.component_id::<T>()
    }

    fn matches_component_set(
        &id: &ComponentId,
        set_contains_id: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        set_contains_id(id)
    }
}

// SAFETY: WorldQuery impl performs no access at all
unsafe impl<T: Component> QueryFilter for With<T> {
    const IS_ARCHETYPAL: bool = true;
    const BATCHABLE: bool = true;

    #[inline(always)]
    unsafe fn filter_fetch(
        _state: &Self::State,
        _fetch: &mut Self::Fetch<'_>,
        _entity: Entity,
        _table_row: TableRow,
    ) -> bool {
        true
    }
}

/// Filter that selects entities without a component `T`.
///
/// This is the negation of [`With`].
///
/// # Examples
///
/// ```
/// # use bevy_ecs::component::Component;
/// # use bevy_ecs::query::Without;
/// # use bevy_ecs::system::IntoSystem;
/// # use bevy_ecs::system::Query;
/// #
/// # #[derive(Component)]
/// # struct Permit;
/// # #[derive(Component)]
/// # struct Name { name: &'static str };
/// #
/// fn no_permit_system(query: Query<&Name, Without<Permit>>) {
///     for name in &query{
///         println!("{} has no permit!", name.name);
///     }
/// }
/// # bevy_ecs::system::assert_is_system(no_permit_system);
/// ```
pub struct Without<T>(PhantomData<T>);

// SAFETY:
// `update_component_access` does not add any accesses.
// This is sound because [`QueryFilter::filter_fetch`] does not access any components.
// `update_component_access` adds a `Without` filter for `T`.
// This is sound because `matches_component_set` returns whether the set does not contain the component.
unsafe impl<T: Component> WorldQuery for Without<T> {
    type Fetch<'w> = ();
    type State = ComponentId;

    fn shrink_fetch<'wlong: 'wshort, 'wshort>(_: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {}

    #[inline]
    unsafe fn init_fetch(
        _world: UnsafeWorldCell,
        _state: &ComponentId,
        _last_run: Tick,
        _this_run: Tick,
    ) {
    }

    const IS_DENSE: bool = {
        match T::STORAGE_TYPE {
            StorageType::Table => true,
            StorageType::SparseSet => false,
        }
    };

    #[inline]
    unsafe fn set_archetype(
        _fetch: &mut (),
        _state: &ComponentId,
        _archetype: &Archetype,
        _table: &Table,
    ) {
    }

    #[inline]
    unsafe fn set_table(_fetch: &mut (), _state: &Self::State, _table: &Table) {}

    #[inline]
    fn update_component_access(&id: &ComponentId, access: &mut FilteredAccess) {
        access.and_without(id);
    }

    fn init_state(world: &mut World) -> ComponentId {
        world.register_component::<T>()
    }

    fn get_state(components: &Components) -> Option<Self::State> {
        components.component_id::<T>()
    }

    fn matches_component_set(
        &id: &ComponentId,
        set_contains_id: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        !set_contains_id(id)
    }
}

// SAFETY: WorldQuery impl performs no access at all
unsafe impl<T: Component> QueryFilter for Without<T> {
    const IS_ARCHETYPAL: bool = true;
    const BATCHABLE: bool = true;

    #[inline(always)]
    unsafe fn filter_fetch(
        _state: &Self::State,
        _fetch: &mut Self::Fetch<'_>,
        _entity: Entity,
        _table_row: TableRow,
    ) -> bool {
        true
    }
}

/// A filter that tests if any of the given filters apply.
///
/// This is useful for example if a system with multiple components in a query only wants to run
/// when one or more of the components have changed.
///
/// The `And` equivalent to this filter is a [`prim@tuple`] testing that all the contained filters
/// apply instead.
///
/// # Examples
///
/// ```
/// # use bevy_ecs::component::Component;
/// # use bevy_ecs::entity::Entity;
/// # use bevy_ecs::query::Changed;
/// # use bevy_ecs::query::Or;
/// # use bevy_ecs::system::IntoSystem;
/// # use bevy_ecs::system::Query;
/// #
/// # #[derive(Component, Debug)]
/// # struct Color {};
/// # #[derive(Component)]
/// # struct Node {};
/// #
/// fn print_cool_entity_system(query: Query<Entity, Or<(Changed<Color>, Changed<Node>)>>) {
///     for entity in &query {
///         println!("Entity {} got a new style or color", entity);
///     }
/// }
/// # bevy_ecs::system::assert_is_system(print_cool_entity_system);
/// ```
pub struct Or<T>(PhantomData<T>);

#[doc(hidden)]
pub struct OrFetch<'w, T: WorldQuery> {
    fetch: T::Fetch<'w>,
    matches: bool,
}

impl<T: WorldQuery> Clone for OrFetch<'_, T> {
    fn clone(&self) -> Self {
        Self {
            fetch: self.fetch.clone(),
            matches: self.matches,
        }
    }
}

macro_rules! impl_or_query_filter {
    ($(#[$meta:meta])* $(($filter: ident, $state: ident)),*) => {
        $(#[$meta])*
        #[expect(
            clippy::allow_attributes,
            reason = "This is a tuple-related macro; as such the lints below may not always apply."
        )]
        #[allow(
            non_snake_case,
            reason = "The names of some variables are provided by the macro's caller, not by us."
        )]
        #[allow(
            unused_variables,
            reason = "Zero-length tuples won't use any of the parameters."
        )]
        #[allow(
            clippy::unused_unit,
            reason = "Zero-length tuples will generate some function bodies equivalent to `()`; however, this macro is meant for all applicable tuples, and as such it makes no sense to rewrite it just for that case."
        )]
        // SAFETY:
        // [`QueryFilter::filter_fetch`] accesses are a subset of the subqueries' accesses
        // This is sound because `update_component_access` adds accesses according to the implementations of all the subqueries.
        // `update_component_access` replace the filters with a disjunction where every element is a conjunction of the previous filters and the filters of one of the subqueries.
        // This is sound because `matches_component_set` returns a disjunction of the results of the subqueries' implementations.
        unsafe impl<$($filter: QueryFilter),*> WorldQuery for Or<($($filter,)*)> {
            type Fetch<'w> = ($(OrFetch<'w, $filter>,)*);
            type State = ($($filter::State,)*);

            fn shrink_fetch<'wlong: 'wshort, 'wshort>(fetch: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {
                let ($($filter,)*) = fetch;
                ($(
                    OrFetch {
                        fetch: $filter::shrink_fetch($filter.fetch),
                        matches: $filter.matches
                    },
                )*)
            }

            const IS_DENSE: bool = true $(&& $filter::IS_DENSE)*;

            #[inline]
            unsafe fn init_fetch<'w, 's>(world: UnsafeWorldCell<'w>, state: &'s Self::State, last_run: Tick, this_run: Tick) -> Self::Fetch<'w> {
                let ($($filter,)*) = state;
                ($(OrFetch {
                    // SAFETY: The invariants are upheld by the caller.
                    fetch: unsafe { $filter::init_fetch(world, $filter, last_run, this_run) },
                    matches: false,
                },)*)
            }

            #[inline]
            unsafe fn set_table<'w, 's>(fetch: &mut Self::Fetch<'w>, state: &'s Self::State, table: &'w Table) {
                // If this is an archetypal query, then it is guaranteed to match all entities,
                // so `filter_fetch` will ignore `$filter.matches` and we don't need to initialize it.
                if Self::IS_ARCHETYPAL {
                    return;
                }
                let ($($filter,)*) = fetch;
                let ($($state,)*) = state;
                $(
                    $filter.matches = $filter::matches_component_set($state, &|id| table.has_column(id));
                    if $filter.matches {
                        // SAFETY: The invariants are upheld by the caller.
                        unsafe { $filter::set_table(&mut $filter.fetch, $state, table); }
                    }
                )*
            }

            #[inline]
            unsafe fn set_archetype<'w, 's>(
                fetch: &mut Self::Fetch<'w>,
                state: &'s Self::State,
                archetype: &'w Archetype,
                table: &'w Table
            ) {
                // If this is an archetypal query, then it is guaranteed to match all entities,
                // so `filter_fetch` will ignore `$filter.matches` and we don't need to initialize it.
                if Self::IS_ARCHETYPAL {
                    return;
                }
                let ($($filter,)*) = fetch;
                let ($($state,)*) = &state;
                $(
                    $filter.matches = $filter::matches_component_set($state, &|id| archetype.contains(id));
                    if $filter.matches {
                        // SAFETY: The invariants are upheld by the caller.
                       unsafe { $filter::set_archetype(&mut $filter.fetch, $state, archetype, table); }
                    }
                )*
            }

            fn update_component_access(state: &Self::State, access: &mut FilteredAccess) {
                let ($($filter,)*) = state;

                let mut new_access = FilteredAccess::matches_nothing();

                $(
                    // Create an intermediate because `access`'s value needs to be preserved
                    // for the next filter, and `_new_access` has to be modified only by `append_or` to it.
                    let mut intermediate = access.clone();
                    $filter::update_component_access($filter, &mut intermediate);
                    new_access.append_or(&intermediate);
                    // Also extend the accesses required to compute the filter. This is required because
                    // otherwise a `Query<(), Or<(Changed<Foo>,)>` won't conflict with `Query<&mut Foo>`.
                    new_access.extend_access(&intermediate);
                )*

                // The required components remain the same as the original `access`.
                new_access.required = core::mem::take(&mut access.required);

                *access = new_access;
            }

            fn init_nested_access(
                state: &Self::State,
                _system_name: Option<&str>,
                _component_access_set: &mut FilteredAccessSet,
                _world: UnsafeWorldCell,
            ) {
                let ($($state,)*) = state;
                $($filter::init_nested_access($state, _system_name, _component_access_set, _world);)*
            }

            fn init_state(world: &mut World) -> Self::State {
                ($($filter::init_state(world),)*)
            }

            fn get_state(components: &Components) -> Option<Self::State> {
                Some(($($filter::get_state(components)?,)*))
            }

            fn matches_component_set(state: &Self::State, set_contains_id: &impl Fn(ComponentId) -> bool) -> bool {
                let ($($filter,)*) = state;
                false $(|| $filter::matches_component_set($filter, set_contains_id))*
            }

            fn update_archetypes(state: &mut Self::State, _world: UnsafeWorldCell) {
                let ($($filter,)*) = state;
                $($filter::update_archetypes($filter, _world);)*
            }
        }

        #[expect(
            clippy::allow_attributes,
            reason = "This is a tuple-related macro; as such the lints below may not always apply."
        )]
        #[allow(
            non_snake_case,
            reason = "The names of some variables are provided by the macro's caller, not by us."
        )]
        #[allow(
            unused_variables,
            reason = "Zero-length tuples won't use any of the parameters."
        )]
        $(#[$meta])*
        // SAFETY: This only performs access that subqueries perform, and they impl `QueryFilter` and so perform no mutable access.
        unsafe impl<$($filter: QueryFilter),*> QueryFilter for Or<($($filter,)*)> {
            const IS_ARCHETYPAL: bool = true $(&& $filter::IS_ARCHETYPAL)*;
            const BATCHABLE: bool = true $(&& $filter::BATCHABLE)*;

            #[inline(always)]
            unsafe fn filter_fetch(
                state: &Self::State,
                fetch: &mut Self::Fetch<'_>,
                entity: Entity,
                table_row: TableRow
            ) -> bool {
                let ($($state,)*) = state;
                let ($($filter,)*) = fetch;
                // If this is an archetypal query, then it is guaranteed to return true,
                // and we can help the compiler remove branches by checking the const `IS_ARCHETYPAL` first.
                (Self::IS_ARCHETYPAL
                    // SAFETY: The invariants are upheld by the caller.
                    $(|| ($filter.matches && unsafe { $filter::filter_fetch($state, &mut $filter.fetch, entity, table_row) }))*
                    // If *none* of the subqueries matched the archetype, then this archetype was added in a transmute.
                    // We must treat those as matching in order to be consistent with `size_hint` for archetypal queries,
                    // so we treat them as matching for non-archetypal queries, as well.
                    || !(false $(|| $filter.matches)*))
            }

            #[inline(always)]
            unsafe fn filter_fetch_mask(
                state: &Self::State,
                fetch: &mut Self::Fetch<'_>,
                entities: &[Entity],
                first_row: u32,
            ) -> u64 {
                if !Self::BATCHABLE {
                    // Preserve the exact per-row short-circuiting call pattern when any
                    // subquery opted out of batching. In-tree iteration never calls this
                    // method when `!BATCHABLE` (it is gated in `fold_over_table_range`);
                    // this branch keeps the documented contract for direct callers.
                    // SAFETY: The invariants are upheld by the caller.
                    return unsafe { filter_fetch_mask_fallback::<Self>(state, fetch, entities, first_row) };
                }
                if Self::IS_ARCHETYPAL {
                    return u64::MAX;
                }
                let ($($state,)*) = state;
                let ($($filter,)*) = fetch;
                let mut _mask = 0u64;
                let mut _any_matches = false;
                $(
                    // Once the mask is saturated no later subquery can change the result;
                    // a saturated mask also implies an earlier subquery already matched,
                    // so `_any_matches` stays correct.
                    if _mask != u64::MAX && $filter.matches {
                        _any_matches = true;
                        // SAFETY: The invariants are upheld by the caller.
                        _mask |= unsafe { $filter::filter_fetch_mask($state, &mut $filter.fetch, entities, first_row) };
                    }
                )*
                if _any_matches {
                    _mask
                } else {
                    // No subquery matched the archetype: it was added via a transmute and is
                    // treated as matching, consistent with `filter_fetch`.
                    u64::MAX
                }
            }
        }
    };
}

macro_rules! impl_tuple_query_filter {
    ($(#[$meta:meta])* $(($name: ident, $state: ident)),*) => {
        #[expect(
            clippy::allow_attributes,
            reason = "This is a tuple-related macro; as such the lints below may not always apply."
        )]
        #[allow(
            non_snake_case,
            reason = "The names of some variables are provided by the macro's caller, not by us."
        )]
        #[allow(
            unused_variables,
            reason = "Zero-length tuples won't use any of the parameters."
        )]
        $(#[$meta])*
        // SAFETY: This only performs access that subqueries perform, and they impl `QueryFilter` and so perform no mutable access.
        unsafe impl<$($name: QueryFilter),*> QueryFilter for ($($name,)*) {
            const IS_ARCHETYPAL: bool = true $(&& $name::IS_ARCHETYPAL)*;
            const BATCHABLE: bool = true $(&& $name::BATCHABLE)*;

            #[inline(always)]
            unsafe fn filter_fetch(
                state: &Self::State,
                fetch: &mut Self::Fetch<'_>,
                entity: Entity,
                table_row: TableRow
            ) -> bool {
                let ($($state,)*) = state;
                let ($($name,)*) = fetch;
                // SAFETY: The invariants are upheld by the caller.
                true $(&& unsafe { $name::filter_fetch($state, $name, entity, table_row) })*
            }

            #[inline(always)]
            unsafe fn filter_fetch_mask(
                state: &Self::State,
                fetch: &mut Self::Fetch<'_>,
                entities: &[Entity],
                first_row: u32,
            ) -> u64 {
                if !Self::BATCHABLE {
                    // Preserve the exact per-row short-circuiting call pattern when any
                    // member opted out of batching. In-tree iteration never calls this
                    // method when `!BATCHABLE` (it is gated in `fold_over_table_range`);
                    // this branch keeps the documented contract for direct callers.
                    // SAFETY: The invariants are upheld by the caller.
                    return unsafe { filter_fetch_mask_fallback::<Self>(state, fetch, entities, first_row) };
                }
                let ($($state,)*) = state;
                let ($($name,)*) = fetch;
                let mut _mask = u64::MAX;
                $(
                    if _mask != 0 {
                        // SAFETY: The invariants are upheld by the caller.
                        _mask &= unsafe { $name::filter_fetch_mask($state, $name, entities, first_row) };
                    }
                )*
                _mask
            }
        }
    };
}

all_tuples!(
    #[doc(fake_variadic)]
    impl_tuple_query_filter,
    0,
    15,
    F,
    S
);
all_tuples!(
    #[doc(fake_variadic)]
    impl_or_query_filter,
    0,
    15,
    F,
    S
);

/// Allows a query to contain entities with the component `T`, bypassing [`DefaultQueryFilters`].
///
/// [`DefaultQueryFilters`]: crate::entity_disabling::DefaultQueryFilters
pub struct Allow<T>(PhantomData<T>);

// SAFETY:
// `update_component_access` does not add any accesses.
// This is sound because [`QueryFilter::filter_fetch`] does not access any components.
// `update_component_access` adds an archetypal filter for `T`.
// This is sound because it doesn't affect the query
unsafe impl<T: Component> WorldQuery for Allow<T> {
    type Fetch<'w> = ();
    type State = ComponentId;

    fn shrink_fetch<'wlong: 'wshort, 'wshort>(_: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {}

    #[inline]
    unsafe fn init_fetch(_: UnsafeWorldCell, _: &ComponentId, _: Tick, _: Tick) {}

    // Even if the component is sparse, this implementation doesn't do anything with it
    const IS_DENSE: bool = true;

    #[inline]
    unsafe fn set_archetype(_: &mut (), _: &ComponentId, _: &Archetype, _: &Table) {}

    #[inline]
    unsafe fn set_table(_: &mut (), _: &ComponentId, _: &Table) {}

    #[inline]
    fn update_component_access(&id: &ComponentId, access: &mut FilteredAccess) {
        access.access_mut().add_archetypal(id);
    }

    fn init_state(world: &mut World) -> ComponentId {
        world.register_component::<T>()
    }

    fn get_state(components: &Components) -> Option<Self::State> {
        components.component_id::<T>()
    }

    fn matches_component_set(_: &ComponentId, _: &impl Fn(ComponentId) -> bool) -> bool {
        // Allow<T> always matches
        true
    }
}

// SAFETY: WorldQuery impl performs no access at all
unsafe impl<T: Component> QueryFilter for Allow<T> {
    const IS_ARCHETYPAL: bool = true;
    const BATCHABLE: bool = true;

    #[inline(always)]
    unsafe fn filter_fetch(
        _: &Self::State,
        _: &mut Self::Fetch<'_>,
        _: Entity,
        _: TableRow,
    ) -> bool {
        true
    }
}

/// A filter on a component that only retains results the first time after they have been added.
///
/// A common use for this filter is one-time initialization.
///
/// To retain all results without filtering but still check whether they were added after the
/// system last ran, use [`Ref<T>`](crate::change_detection::Ref).
///
/// **Note** that this includes changes that happened before the first time this `Query` was run.
///
/// # Deferred
///
/// Note, that entity modifications issued with [`Commands`](crate::system::Commands)
/// are visible only after deferred operations are applied, typically after the system
/// that queued them.
///
/// # Time complexity
///
/// `Added` is not [`ArchetypeFilter`], which practically means that
/// if the query (with `T` component filter) matches a million entities,
/// `Added<T>` filter will iterate over all of them even if none of them were just added.
///
/// For example, these two systems are roughly equivalent in terms of performance:
///
/// ```
/// # use bevy_ecs::change_detection::{DetectChanges, Ref};
/// # use bevy_ecs::entity::Entity;
/// # use bevy_ecs::query::Added;
/// # use bevy_ecs::system::Query;
/// # use bevy_ecs_macros::Component;
/// # #[derive(Component)]
/// # struct MyComponent;
/// # #[derive(Component)]
/// # struct Transform;
///
/// fn system1(q: Query<&MyComponent, Added<Transform>>) {
///     for item in &q { /* component added */ }
/// }
///
/// fn system2(q: Query<(&MyComponent, Ref<Transform>)>) {
///     for item in &q {
///         if item.1.is_added() { /* component added */ }
///     }
/// }
/// ```
///
/// # Examples
///
/// ```
/// # use bevy_ecs::component::Component;
/// # use bevy_ecs::query::Added;
/// # use bevy_ecs::system::IntoSystem;
/// # use bevy_ecs::system::Query;
/// #
/// # #[derive(Component, Debug)]
/// # struct Name {};
///
/// fn print_add_name_component(query: Query<&Name, Added<Name>>) {
///     for name in &query {
///         println!("Named entity created: {:?}", name)
///     }
/// }
///
/// # bevy_ecs::system::assert_is_system(print_add_name_component);
/// ```
pub struct Added<T>(PhantomData<T>);

#[doc(hidden)]
pub struct AddedFetch<'w, T: Component> {
    ticks: StorageSwitch<
        T,
        // T::STORAGE_TYPE = StorageType::Table
        Option<ThinSlicePtr<'w, UnsafeCell<Tick>>>,
        // T::STORAGE_TYPE = StorageType::SparseSet
        // Can be `None` when the component has never been inserted
        Option<&'w ComponentSparseSet>,
    >,
    this_run: Tick,
    // Precomputed [`Tick::newness_threshold`]; a tick matches iff
    // [`Tick::is_newer_than_threshold`] against it.
    threshold: u32,
}

impl<T: Component> Clone for AddedFetch<'_, T> {
    fn clone(&self) -> Self {
        Self {
            ticks: self.ticks,
            this_run: self.this_run,
            threshold: self.threshold,
        }
    }
}

// SAFETY:
// [`QueryFilter::filter_fetch`] accesses a single component in a readonly way.
// This is sound because `update_component_access` adds read access for that component and panics when appropriate.
// `update_component_access` adds a `With` filter for a component.
// This is sound because `matches_component_set` returns whether the set contains that component.
unsafe impl<T: Component> WorldQuery for Added<T> {
    type Fetch<'w> = AddedFetch<'w, T>;
    type State = ComponentId;

    fn shrink_fetch<'wlong: 'wshort, 'wshort>(fetch: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {
        fetch
    }

    #[inline]
    unsafe fn init_fetch<'w, 's>(
        world: UnsafeWorldCell<'w>,
        &id: &'s ComponentId,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        Self::Fetch::<'w> {
            ticks: StorageSwitch::new(
                || None,
                || {
                    // SAFETY: The underlying type associated with `component_id` is `T`,
                    // which we are allowed to access since we registered it in `update_component_access`.
                    // Note that we do not actually access any components' ticks in this function, we just get a shared
                    // reference to the sparse set, which is used to access the components' ticks in `Self::fetch`.
                    unsafe { world.storages().sparse_sets.get(id) }
                },
            ),
            this_run,
            threshold: Tick::newness_threshold(last_run, this_run),
        }
    }

    const IS_DENSE: bool = {
        match T::STORAGE_TYPE {
            StorageType::Table => true,
            StorageType::SparseSet => false,
        }
    };

    #[inline]
    unsafe fn set_archetype<'w, 's>(
        fetch: &mut Self::Fetch<'w>,
        component_id: &'s ComponentId,
        _archetype: &'w Archetype,
        table: &'w Table,
    ) {
        if Self::IS_DENSE {
            // SAFETY: `set_archetype`'s safety rules are a super set of the `set_table`'s ones.
            unsafe {
                Self::set_table(fetch, component_id, table);
            }
        }
    }

    #[inline]
    unsafe fn set_table<'w, 's>(
        fetch: &mut Self::Fetch<'w>,
        &component_id: &'s ComponentId,
        table: &'w Table,
    ) {
        let table_ticks = Some(
            table
                .get_added_ticks_slice_for(component_id)
                .debug_checked_unwrap()
                .into(),
        );
        // SAFETY: set_table is only called when T::STORAGE_TYPE = StorageType::Table
        unsafe { fetch.ticks.set_table(table_ticks) };
    }

    #[inline]
    fn update_component_access(&id: &ComponentId, access: &mut FilteredAccess) {
        if access.access().has_write(id) {
            panic!("$state_name<{}> conflicts with a previous access in this query. Shared access cannot coincide with exclusive access.", DebugName::type_name::<T>());
        }
        access.add_read(id);
    }

    fn init_state(world: &mut World) -> ComponentId {
        world.register_component::<T>()
    }

    fn get_state(components: &Components) -> Option<ComponentId> {
        components.component_id::<T>()
    }

    fn matches_component_set(
        &id: &ComponentId,
        set_contains_id: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        set_contains_id(id)
    }
}

// SAFETY: WorldQuery impl performs only read access on ticks
unsafe impl<T: Component> QueryFilter for Added<T> {
    const IS_ARCHETYPAL: bool = false;
    const BATCHABLE: bool = true;

    #[inline(always)]
    unsafe fn filter_fetch(
        _state: &Self::State,
        fetch: &mut Self::Fetch<'_>,
        entity: Entity,
        table_row: TableRow,
    ) -> bool {
        // SAFETY: The invariants are upheld by the caller.
        fetch.ticks.extract(
            |table| {
                // SAFETY: set_table was previously called
                let table = unsafe { table.debug_checked_unwrap() };
                // SAFETY: The caller ensures `table_row` is in range.
                let tick = unsafe { table.get_unchecked(table_row.index()) };

                tick.deref()
                    .is_newer_than_threshold(fetch.this_run, fetch.threshold)
            },
            |sparse_set| {
                // SAFETY: The caller ensures `entity` is in range.
                let tick = unsafe {
                    sparse_set
                        .debug_checked_unwrap()
                        .get_added_tick(entity)
                        .debug_checked_unwrap()
                };

                tick.deref()
                    .is_newer_than_threshold(fetch.this_run, fetch.threshold)
            },
        )
    }

    #[inline]
    unsafe fn filter_fetch_mask(
        _state: &Self::State,
        fetch: &mut Self::Fetch<'_>,
        entities: &[Entity],
        first_row: u32,
    ) -> u64 {
        // SAFETY: The invariants are upheld by the caller.
        unsafe {
            tick_filter_fetch_mask(
                &fetch.ticks,
                fetch.this_run,
                fetch.threshold,
                entities,
                first_row,
                |sparse_set, entity| {
                    // SAFETY: The caller ensures each entity is stored in the current
                    // table, and therefore present in the sparse set.
                    sparse_set
                        .get_added_tick(entity)
                        .debug_checked_unwrap()
                        .read()
                },
            )
        }
    }
}

/// A filter on a component that only retains results the first time after they have been added or mutably dereferenced.
///
/// A common use for this filter is avoiding redundant work when values have not changed.
///
/// **Note** that simply *mutably dereferencing* a component is considered a change ([`DerefMut`](std::ops::DerefMut)).
/// Bevy does not compare components to their previous values.
///
/// To retain all results without filtering but still check whether they were changed after the
/// system last ran, use [`Ref<T>`](crate::change_detection::Ref).
///
/// **Note** that this includes changes that happened before the first time this `Query` was run.
///
/// # Deferred
///
/// Note, that entity modifications issued with [`Commands`](crate::system::Commands)
/// (like entity creation or entity component addition or removal) are visible only
/// after deferred operations are applied, typically after the system that queued them.
///
/// # Time complexity
///
/// `Changed` is not [`ArchetypeFilter`], which practically means that
/// if query (with `T` component filter) matches million entities,
/// `Changed<T>` filter will iterate over all of them even if none of them were changed.
///
/// For example, these two systems are roughly equivalent in terms of performance:
///
/// ```
/// # use bevy_ecs::change_detection::DetectChanges;
/// # use bevy_ecs::entity::Entity;
/// # use bevy_ecs::query::Changed;
/// # use bevy_ecs::system::Query;
/// # use bevy_ecs::world::Ref;
/// # use bevy_ecs_macros::Component;
/// # #[derive(Component)]
/// # struct MyComponent;
/// # #[derive(Component)]
/// # struct Transform;
///
/// fn system1(q: Query<&MyComponent, Changed<Transform>>) {
///     for item in &q { /* component changed */ }
/// }
///
/// fn system2(q: Query<(&MyComponent, Ref<Transform>)>) {
///     for item in &q {
///         if item.1.is_changed() { /* component changed */ }
///     }
/// }
/// ```
///
/// # Examples
///
/// ```
/// # use bevy_ecs::component::Component;
/// # use bevy_ecs::query::Changed;
/// # use bevy_ecs::system::IntoSystem;
/// # use bevy_ecs::system::Query;
/// #
/// # #[derive(Component, Debug)]
/// # struct Name {};
/// # #[derive(Component)]
/// # struct Transform {};
///
/// fn print_moving_objects_system(query: Query<&Name, Changed<Transform>>) {
///     for name in &query {
///         println!("Entity Moved: {:?}", name);
///     }
/// }
///
/// # bevy_ecs::system::assert_is_system(print_moving_objects_system);
/// ```
pub struct Changed<T>(PhantomData<T>);

#[doc(hidden)]
pub struct ChangedFetch<'w, T: Component> {
    ticks: StorageSwitch<
        T,
        Option<ThinSlicePtr<'w, UnsafeCell<Tick>>>,
        // Can be `None` when the component has never been inserted
        Option<&'w ComponentSparseSet>,
    >,
    this_run: Tick,
    // Precomputed [`Tick::newness_threshold`]; a tick matches iff
    // [`Tick::is_newer_than_threshold`] against it.
    threshold: u32,
}

impl<T: Component> Clone for ChangedFetch<'_, T> {
    fn clone(&self) -> Self {
        Self {
            ticks: self.ticks,
            this_run: self.this_run,
            threshold: self.threshold,
        }
    }
}

// SAFETY:
// `fetch` accesses a single component in a readonly way.
// This is sound because `update_component_access` add read access for that component and panics when appropriate.
// `update_component_access` adds a `With` filter for a component.
// This is sound because `matches_component_set` returns whether the set contains that component.
unsafe impl<T: Component> WorldQuery for Changed<T> {
    type Fetch<'w> = ChangedFetch<'w, T>;
    type State = ComponentId;

    fn shrink_fetch<'wlong: 'wshort, 'wshort>(fetch: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {
        fetch
    }

    #[inline]
    unsafe fn init_fetch<'w, 's>(
        world: UnsafeWorldCell<'w>,
        &id: &'s ComponentId,
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        Self::Fetch::<'w> {
            ticks: StorageSwitch::new(
                || None,
                || {
                    // SAFETY: The underlying type associated with `component_id` is `T`,
                    // which we are allowed to access since we registered it in `update_component_access`.
                    // Note that we do not actually access any components' ticks in this function, we just get a shared
                    // reference to the sparse set, which is used to access the components' ticks in `Self::fetch`.
                    unsafe { world.storages().sparse_sets.get(id) }
                },
            ),
            this_run,
            threshold: Tick::newness_threshold(last_run, this_run),
        }
    }

    const IS_DENSE: bool = {
        match T::STORAGE_TYPE {
            StorageType::Table => true,
            StorageType::SparseSet => false,
        }
    };

    #[inline]
    unsafe fn set_archetype<'w, 's>(
        fetch: &mut Self::Fetch<'w>,
        component_id: &'s ComponentId,
        _archetype: &'w Archetype,
        table: &'w Table,
    ) {
        if Self::IS_DENSE {
            // SAFETY: `set_archetype`'s safety rules are a super set of the `set_table`'s ones.
            unsafe {
                Self::set_table(fetch, component_id, table);
            }
        }
    }

    #[inline]
    unsafe fn set_table<'w, 's>(
        fetch: &mut Self::Fetch<'w>,
        &component_id: &'s ComponentId,
        table: &'w Table,
    ) {
        let table_ticks = Some(
            table
                .get_changed_ticks_slice_for(component_id)
                .debug_checked_unwrap()
                .into(),
        );
        // SAFETY: set_table is only called when T::STORAGE_TYPE = StorageType::Table
        unsafe { fetch.ticks.set_table(table_ticks) };
    }

    #[inline]
    fn update_component_access(&id: &ComponentId, access: &mut FilteredAccess) {
        if access.access().has_write(id) {
            panic!("$state_name<{}> conflicts with a previous access in this query. Shared access cannot coincide with exclusive access.", DebugName::type_name::<T>());
        }
        access.add_read(id);
    }

    fn init_state(world: &mut World) -> ComponentId {
        world.register_component::<T>()
    }

    fn get_state(components: &Components) -> Option<ComponentId> {
        components.component_id::<T>()
    }

    fn matches_component_set(
        &id: &ComponentId,
        set_contains_id: &impl Fn(ComponentId) -> bool,
    ) -> bool {
        set_contains_id(id)
    }
}

// SAFETY: WorldQuery impl performs only read access on ticks
unsafe impl<T: Component> QueryFilter for Changed<T> {
    const IS_ARCHETYPAL: bool = false;
    const BATCHABLE: bool = true;

    #[inline(always)]
    unsafe fn filter_fetch(
        _state: &Self::State,
        fetch: &mut Self::Fetch<'_>,
        entity: Entity,
        table_row: TableRow,
    ) -> bool {
        // SAFETY: The invariants are upheld by the caller.
        fetch.ticks.extract(
            |table| {
                // SAFETY: set_table was previously called
                let table = unsafe { table.debug_checked_unwrap() };
                // SAFETY: The caller ensures `table_row` is in range.
                let tick = unsafe { table.get_unchecked(table_row.index()) };

                tick.deref()
                    .is_newer_than_threshold(fetch.this_run, fetch.threshold)
            },
            |sparse_set| {
                // SAFETY: The caller ensures `entity` is in range.
                let tick = unsafe {
                    sparse_set
                        .debug_checked_unwrap()
                        .get_changed_tick(entity)
                        .debug_checked_unwrap()
                };

                tick.deref()
                    .is_newer_than_threshold(fetch.this_run, fetch.threshold)
            },
        )
    }

    #[inline]
    unsafe fn filter_fetch_mask(
        _state: &Self::State,
        fetch: &mut Self::Fetch<'_>,
        entities: &[Entity],
        first_row: u32,
    ) -> u64 {
        // SAFETY: The invariants are upheld by the caller.
        unsafe {
            tick_filter_fetch_mask(
                &fetch.ticks,
                fetch.this_run,
                fetch.threshold,
                entities,
                first_row,
                |sparse_set, entity| {
                    // SAFETY: The caller ensures each entity is stored in the current
                    // table, and therefore present in the sparse set.
                    sparse_set
                        .get_changed_tick(entity)
                        .debug_checked_unwrap()
                        .read()
                },
            )
        }
    }
}

/// A filter that only retains results the first time after the entity has been spawned.
///
/// A common use for this filter is one-time initialization.
///
/// To retain all results without filtering but still check whether they were spawned after the
/// system last ran, use [`SpawnDetails`](crate::query::SpawnDetails) instead.
///
/// **Note** that this includes entities that spawned before the first time this Query was run.
///
/// # Deferred
///
/// Note, that entity spawns issued with [`Commands`](crate::system::Commands)
/// are visible only after deferred operations are applied, typically after the
/// system that queued them.
///
/// # Time complexity
///
/// `Spawned` is not [`ArchetypeFilter`], which practically means that if query matches million
/// entities, `Spawned` filter will iterate over all of them even if none of them were spawned.
///
/// For example, these two systems are roughly equivalent in terms of performance:
///
/// ```
/// # use bevy_ecs::entity::Entity;
/// # use bevy_ecs::system::Query;
/// # use bevy_ecs::query::Spawned;
/// # use bevy_ecs::query::SpawnDetails;
///
/// fn system1(query: Query<Entity, Spawned>) {
///     for entity in &query { /* entity spawned */ }
/// }
///
/// fn system2(query: Query<(Entity, SpawnDetails)>) {
///     for (entity, spawned) in &query {
///         if spawned.is_spawned() { /* entity spawned */ }
///     }
/// }
/// ```
///
/// # Examples
///
/// ```
/// # use bevy_ecs::component::Component;
/// # use bevy_ecs::query::Spawned;
/// # use bevy_ecs::system::IntoSystem;
/// # use bevy_ecs::system::Query;
/// #
/// # #[derive(Component, Debug)]
/// # struct Name {};
///
/// fn print_spawning_entities(query: Query<&Name, Spawned>) {
///     for name in &query {
///         println!("Entity spawned: {:?}", name);
///     }
/// }
///
/// # bevy_ecs::system::assert_is_system(print_spawning_entities);
/// ```
pub struct Spawned;

#[doc(hidden)]
#[derive(Clone)]
pub struct SpawnedFetch<'w> {
    entities: &'w Entities,
    last_run: Tick,
    this_run: Tick,
}

// SAFETY: WorldQuery impl accesses no components or component ticks
unsafe impl WorldQuery for Spawned {
    type Fetch<'w> = SpawnedFetch<'w>;
    type State = ();

    fn shrink_fetch<'wlong: 'wshort, 'wshort>(fetch: Self::Fetch<'wlong>) -> Self::Fetch<'wshort> {
        fetch
    }

    #[inline]
    unsafe fn init_fetch<'w, 's>(
        world: UnsafeWorldCell<'w>,
        _state: &'s (),
        last_run: Tick,
        this_run: Tick,
    ) -> Self::Fetch<'w> {
        SpawnedFetch {
            entities: world.entities(),
            last_run,
            this_run,
        }
    }

    const IS_DENSE: bool = true;

    #[inline]
    unsafe fn set_archetype<'w, 's>(
        _fetch: &mut Self::Fetch<'w>,
        _state: &'s (),
        _archetype: &'w Archetype,
        _table: &'w Table,
    ) {
    }

    #[inline]
    unsafe fn set_table<'w, 's>(_fetch: &mut Self::Fetch<'w>, _state: &'s (), _table: &'w Table) {}

    #[inline]
    fn update_component_access(_state: &(), _access: &mut FilteredAccess) {}

    fn init_state(_world: &mut World) {}

    fn get_state(_components: &Components) -> Option<()> {
        Some(())
    }

    fn matches_component_set(_state: &(), _set_contains_id: &impl Fn(ComponentId) -> bool) -> bool {
        true
    }
}

// SAFETY: WorldQuery impl accesses no components or component ticks
unsafe impl QueryFilter for Spawned {
    const IS_ARCHETYPAL: bool = false;
    // Pure read of the entity's spawn tick; the default (scalar) `filter_fetch_mask`
    // applies, but this must not block batching for filters composed with it.
    const BATCHABLE: bool = true;

    #[inline(always)]
    unsafe fn filter_fetch(
        _state: &Self::State,
        fetch: &mut Self::Fetch<'_>,
        entity: Entity,
        _table_row: TableRow,
    ) -> bool {
        // SAFETY: only living entities are queried
        let spawned = unsafe {
            fetch
                .entities
                .entity_get_spawned_or_despawned_unchecked(entity)
                .1
        };
        spawned.is_newer_than(fetch.last_run, fetch.this_run)
    }
}

/// A marker trait to indicate that the filter works at an archetype level.
///
/// This is needed to:
/// - implement [`ExactSizeIterator`] for [`QueryIter`](crate::query::QueryIter) that contains archetype-level filters.
/// - ensure table filtering for [`QueryContiguousIter`](crate::query::QueryContiguousIter).
///
/// The trait must only be implemented for filters where its corresponding [`QueryFilter::IS_ARCHETYPAL`]
/// is [`prim@true`]. As such, only the [`With`] and [`Without`] filters can implement the trait.
/// [Tuples](prim@tuple) and [`Or`] filters are automatically implemented with the trait only if its containing types
/// also implement the same trait.
///
/// [`Added`], [`Changed`] and [`Spawned`] work with entities, and therefore are not archetypal. As such
/// they do not implement [`ArchetypeFilter`].
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid `Query` filter based on archetype information",
    label = "invalid `Query` filter",
    note = "an `ArchetypeFilter` typically uses a combination of `With<T>` and `Without<T>` statements"
)]
pub trait ArchetypeFilter: QueryFilter {}

impl<T: Component> ArchetypeFilter for With<T> {}

impl<T: Component> ArchetypeFilter for Without<T> {}

macro_rules! impl_archetype_filter_tuple {
    ($(#[$meta:meta])* $($filter: ident),*) => {
        $(#[$meta])*
        impl<$($filter: ArchetypeFilter),*> ArchetypeFilter for ($($filter,)*) {}
    };
}

macro_rules! impl_archetype_or_filter_tuple {
    ($(#[$meta:meta])* $($filter: ident),*) => {
        $(#[$meta])*
        impl<$($filter: ArchetypeFilter),*> ArchetypeFilter for Or<($($filter,)*)> {}
    };
}

all_tuples!(
    #[doc(fake_variadic)]
    impl_archetype_filter_tuple,
    0,
    15,
    F
);

all_tuples!(
    #[doc(fake_variadic)]
    impl_archetype_or_filter_tuple,
    0,
    15,
    F
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::change_detection::MAX_CHANGE_AGE;
    use alloc::{vec, vec::Vec};

    #[derive(Component)]
    struct TableA(u32);

    #[derive(Component)]
    struct TableB(u32);

    #[derive(Component)]
    #[component(storage = "SparseSet")]
    struct SparseC(u32);

    #[test]
    fn threshold_predicate_equivalent_to_is_newer_than() {
        let offsets = [
            0,
            1,
            2,
            63,
            64,
            65,
            100,
            MAX_CHANGE_AGE - 1,
            MAX_CHANGE_AGE,
            MAX_CHANGE_AGE + 1,
            u32::MAX - 1,
            u32::MAX,
        ];
        for &now in &offsets {
            let this_run = Tick::new(now);
            for &system_age in &offsets {
                let last_run = Tick::new(now.wrapping_sub(system_age));
                let threshold = Tick::newness_threshold(last_run, this_run);
                for &tick_age in &offsets {
                    let tick = Tick::new(now.wrapping_sub(tick_age));
                    assert_eq!(
                        tick.is_newer_than_threshold(this_run, threshold),
                        tick.is_newer_than(last_run, this_run),
                        "now={now} system_age={system_age} tick_age={tick_age}"
                    );
                }
            }
        }
    }

    /// Asserts that the `next()` path (plain iteration) and the fold path (which uses
    /// `filter_fetch_mask` blocks for batchable filters) agree with each other and with
    /// `expected` (as a set).
    fn assert_next_and_fold_agree<F: QueryFilter>(world: &mut World, expected: &mut Vec<Entity>) {
        let mut query = world.query_filtered::<Entity, F>();
        let via_next: Vec<Entity> = query.iter(world).collect();
        let mut via_fold: Vec<Entity> = Vec::new();
        query.iter(world).for_each(|entity| via_fold.push(entity));
        assert_eq!(via_next, via_fold);
        let mut sorted = via_next;
        sorted.sort();
        expected.sort();
        assert_eq!(&sorted, expected);
    }

    #[test]
    fn batched_changed_matches_scalar_across_blocks() {
        let mut world = World::new();
        // Two archetypes so multiple tables (and partial trailing blocks) are exercised.
        let entities: Vec<Entity> = (0..200u32)
            .map(|i| {
                if i % 3 == 0 {
                    world.spawn((TableA(i), TableB(i))).id()
                } else {
                    world.spawn(TableA(i)).id()
                }
            })
            .collect();
        world.clear_trackers();

        // Scattered singles, a >64-row run crossing block boundaries, and the last row.
        let mut expected = Vec::new();
        for (i, &entity) in entities.iter().enumerate() {
            if i % 7 == 0 || (60..=130).contains(&i) || i == 199 {
                world.get_mut::<TableA>(entity).unwrap().0 ^= 1;
                expected.push(entity);
            }
        }
        assert_next_and_fold_agree::<Changed<TableA>>(&mut world, &mut expected);
    }

    #[test]
    fn batched_changed_none_and_all() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..150u32).map(|i| world.spawn(TableA(i)).id()).collect();
        world.clear_trackers();
        assert_next_and_fold_agree::<Changed<TableA>>(&mut world, &mut vec![]);

        for &entity in &entities {
            world.get_mut::<TableA>(entity).unwrap().0 ^= 1;
        }
        let mut expected = entities;
        assert_next_and_fold_agree::<Changed<TableA>>(&mut world, &mut expected);
    }

    #[test]
    fn batched_added_matches_scalar() {
        let mut world = World::new();
        for i in 0..100u32 {
            world.spawn(TableA(i));
        }
        world.clear_trackers();
        let mut expected: Vec<Entity> = (0..70u32).map(|i| world.spawn(TableA(i)).id()).collect();
        assert_next_and_fold_agree::<Added<TableA>>(&mut world, &mut expected);
    }

    #[test]
    fn batched_tuple_and_or_match_scalar() {
        let mut world = World::new();
        let mut spawned = Vec::new();
        for i in 0..150u32 {
            if i % 2 == 0 {
                spawned.push((world.spawn((TableA(i), TableB(i))).id(), true));
            } else {
                spawned.push((world.spawn(TableA(i)).id(), false));
            }
        }
        world.clear_trackers();

        let mut changed_a = Vec::new();
        let mut changed_b = Vec::new();
        for (i, &(entity, has_b)) in spawned.iter().enumerate() {
            if i % 5 == 0 {
                world.get_mut::<TableA>(entity).unwrap().0 ^= 1;
                changed_a.push(entity);
            }
            if has_b && i % 4 == 0 {
                world.get_mut::<TableB>(entity).unwrap().0 ^= 1;
                changed_b.push(entity);
            }
        }

        let mut expected_tuple: Vec<Entity> = spawned
            .iter()
            .filter(|(entity, has_b)| *has_b && changed_a.contains(entity))
            .map(|&(entity, _)| entity)
            .collect();
        assert_next_and_fold_agree::<(Changed<TableA>, With<TableB>)>(
            &mut world,
            &mut expected_tuple,
        );

        let mut expected_or: Vec<Entity> = spawned
            .iter()
            .filter(|(entity, _)| changed_a.contains(entity) || changed_b.contains(entity))
            .map(|&(entity, _)| entity)
            .collect();
        assert_next_and_fold_agree::<Or<(Changed<TableA>, Changed<TableB>)>>(
            &mut world,
            &mut expected_or,
        );
    }

    #[test]
    fn batched_changed_sparse_matches_scalar() {
        let mut world = World::new();
        let entities: Vec<Entity> = (0..100u32).map(|i| world.spawn(SparseC(i)).id()).collect();
        world.clear_trackers();
        let mut expected = Vec::new();
        for (i, &entity) in entities.iter().enumerate() {
            if i % 4 == 0 {
                world.get_mut::<SparseC>(entity).unwrap().0 ^= 1;
                expected.push(entity);
            }
        }
        assert_next_and_fold_agree::<Changed<SparseC>>(&mut world, &mut expected);
    }

    #[test]
    fn batched_or_transmute_fallback_matches_scalar() {
        let mut world = World::new();
        world.register_component::<TableA>();
        for i in 0..70u32 {
            world.spawn(TableB(i));
        }
        world.clear_trackers();

        // `TableB`-only archetype: no `Or` branch matches it, so every row must match via
        // the transmute fallback, on both the `next()` and the fold (mask) path.
        let mut query = world
            .query::<(&TableB, Option<&TableA>)>()
            .transmute_filtered::<Entity, Or<(Changed<TableA>,)>>(&world);
        let via_next: Vec<Entity> = query.iter(&world).collect();
        let mut via_fold: Vec<Entity> = Vec::new();
        query.iter(&world).for_each(|entity| via_fold.push(entity));
        assert_eq!(via_next, via_fold);
        assert_eq!(via_next.len(), 70);
    }
}
