use bevy_platform::sync::atomic::AtomicU64;
use bevy_platform::sync::atomic::Ordering;

/// This is used by the [`bsn!`](crate::bsn) macro to generate compile-time only references to symbols. Currently this is used
/// to add IDE support for nested type names, as it allows us to pass the input Ident from the input to the output code.
pub const fn touch_type<T>() {}

/// Creates a tuple that will be nested after it passes 11 items.
/// When there is a single item, it is _not_ wrapped in a tuple.
/// This is implemented in a way that creates the smallest number of trait impls possible.
#[macro_export]
#[doc(hidden)]
macro_rules! auto_nest_tuple {
    // direct expansion
    () => { () };
    ($a:expr) => {
        $a
    };
    ($a:expr, $b:expr) => {
        (
            $a,
            $b,
        )
    };
    ($a:expr, $b:expr, $c:expr) => {
        (
            $a,
            $b,
            $c,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
            $f,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
            $f,
            $g,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
            $f,
            $g,
            $h,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
            $f,
            $g,
            $h,
            $i,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
            $f,
            $g,
            $h,
            $i,
            $j,
        )
    };
    ($a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr, $g:expr, $h:expr, $i:expr, $j:expr, $k:expr) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
            $f,
            $g,
            $h,
            $i,
            $j,
            $k,
        )
    };

    // recursive expansion
    (
        $a:expr, $b:expr, $c:expr, $d:expr, $e:expr, $f:expr,
        $g:expr, $h:expr, $i:expr, $j:expr, $k:expr, $($rest:expr),*
    ) => {
        (
            $a,
            $b,
            $c,
            $d,
            $e,
            $f,
            $g,
            $h,
            $i,
            $j,
            $k,
            $crate::auto_nest_tuple!($($rest),*)
        )
    };
}

/// This is used by the [`bsn!`](crate::bsn) derive to work around [this Rust limitation](https://github.com/rust-lang/rust/issues/86935).
/// A fix is implemented and on track for stabilization. If it is ever implemented, we can remove this.
pub type PathResolveHelper<T> = T;

/// Global monotonic counter backing [`next_scope_id`].
static BSN_SCOPE_ID: AtomicU64 = AtomicU64::new(0);

/// Returns a fresh, globally-unique scope id.
///
/// The [`bsn!`](crate::bsn) macro calls this once per evaluation of a scene that uses named entity
/// references, so the same `#Name` evaluated many times (e.g. in a loop) resolves to a distinct
/// entity each time.
#[inline]
pub fn next_scope_id() -> u64 {
    BSN_SCOPE_ID.fetch_add(1, Ordering::Relaxed)
}
