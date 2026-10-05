use crate::io::{
    get_meta_path, AssetReader, AssetReaderError, AssetWriter, AssetWriterError, PathStream,
    Reader, ReaderNotSeekableError, SeekableReader, StackFuture, VecReader, Writer,
    STACK_FUTURE_SIZE,
};
use async_fs::{read_dir, File};
#[cfg(not(target_os = "windows"))]
use async_io::Timer;
#[cfg(not(target_os = "windows"))]
use async_lock::{Semaphore, SemaphoreGuard};
use blocking::{unblock, Unblock};
use futures_io::{AsyncRead, AsyncSeek};
use futures_lite::StreamExt;

use alloc::{borrow::ToOwned, boxed::Box, vec::Vec};
#[cfg(target_os = "windows")]
use core::marker::PhantomData;
#[cfg(not(target_os = "windows"))]
use core::time::Duration;
use core::{
    pin::Pin,
    task::{Context, Poll},
};
#[cfg(not(target_os = "windows"))]
use futures_util::{future, pin_mut};
use std::{
    io::{Read, SeekFrom},
    path::{Path, PathBuf},
};

use super::{FileAssetReader, FileAssetWriter};

impl Reader for File {
    fn seekable(&mut self) -> Result<&mut dyn SeekableReader, ReaderNotSeekableError> {
        Ok(self)
    }
}

/// Files up to this many bytes are read into memory in a single blocking call. Larger files are
/// streamed.
///
/// This equals the default pipe capacity of [`Unblock`], which a streamed read reserves on its
/// first poll, so a whole-file buffer never exceeds what streaming the same file would reserve.
const WHOLE_FILE_READ_LIMIT: u64 = 8 * 1024 * 1024;

/// Pipe capacity of the [`Unblock`] wrapping each file opened for writing.
const WRITE_PIPE_CAPACITY: usize = 64 * 1024;

// Set to OS default limit / 2
// macos & ios: 256
// linux & android: 1024
#[cfg(any(target_os = "macos", target_os = "ios"))]
static OPEN_FILE_LIMITER: Semaphore = Semaphore::new(128);
#[cfg(not(any(target_os = "macos", target_os = "ios", target_os = "windows")))]
static OPEN_FILE_LIMITER: Semaphore = Semaphore::new(512);

#[cfg(not(target_os = "windows"))]
async fn maybe_get_semaphore<'a>() -> Option<SemaphoreGuard<'a>> {
    let guard_future = OPEN_FILE_LIMITER.acquire();
    let timeout_future = Timer::after(Duration::from_millis(500));
    pin_mut!(guard_future);
    pin_mut!(timeout_future);

    match future::select(guard_future, timeout_future).await {
        future::Either::Left((guard, _)) => Some(guard),
        future::Either::Right((_, _)) => None,
    }
}

struct GuardedFile<'a> {
    file: File,
    #[cfg(not(target_os = "windows"))]
    _guard: Option<SemaphoreGuard<'a>>,
    #[cfg(target_os = "windows")]
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> AsyncRead for GuardedFile<'a> {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        Pin::new(&mut self.file).poll_read(cx, buf)
    }
}

impl<'a> AsyncSeek for GuardedFile<'a> {
    fn poll_seek(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        pos: SeekFrom,
    ) -> Poll<std::io::Result<u64>> {
        Pin::new(&mut self.file).poll_seek(cx, pos)
    }
}

/// A file opened on a blocking thread by [`open_for_read`].
enum Opened {
    /// The entire contents of a file no larger than [`WHOLE_FILE_READ_LIMIT`].
    Whole(Vec<u8>),
    /// A file larger than [`WHOLE_FILE_READ_LIMIT`], with its length.
    Stream { file: std::fs::File, len: u64 },
}

/// Opens `path`, reading it whole if it is no larger than [`WHOLE_FILE_READ_LIMIT`].
fn open_for_read(path: &Path) -> std::io::Result<Opened> {
    let mut file = std::fs::File::open(path)?;
    let len = file.metadata()?.len();
    if len > WHOLE_FILE_READ_LIMIT {
        return Ok(Opened::Stream { file, len });
    }
    // `len` is at most `WHOLE_FILE_READ_LIMIT`, so it fits in `usize`.
    let mut bytes = Vec::with_capacity(len as usize);
    file.read_to_end(&mut bytes)?;
    Ok(Opened::Whole(bytes))
}

fn map_not_found(error: std::io::Error, full_path: PathBuf) -> AssetReaderError {
    if error.kind() == std::io::ErrorKind::NotFound {
        AssetReaderError::NotFound(full_path)
    } else {
        error.into()
    }
}

/// A [`Reader`] over a file in the local filesystem.
///
/// Files no larger than [`WHOLE_FILE_READ_LIMIT`] are held in memory; larger files are streamed
/// from an open handle.
enum FileReader<'a> {
    Whole(VecReader),
    Stream { file: GuardedFile<'a>, len: u64 },
}

impl AsyncRead for FileReader<'_> {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<std::io::Result<usize>> {
        match self.get_mut() {
            FileReader::Whole(reader) => Pin::new(reader).poll_read(cx, buf),
            FileReader::Stream { file, .. } => Pin::new(file).poll_read(cx, buf),
        }
    }
}

impl AsyncSeek for FileReader<'_> {
    fn poll_seek(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        pos: SeekFrom,
    ) -> Poll<std::io::Result<u64>> {
        match self.get_mut() {
            FileReader::Whole(reader) => Pin::new(reader).poll_seek(cx, pos),
            FileReader::Stream { file, .. } => Pin::new(file).poll_seek(cx, pos),
        }
    }
}

impl Reader for FileReader<'_> {
    fn read_to_end<'a>(
        &'a mut self,
        buf: &'a mut Vec<u8>,
    ) -> StackFuture<'a, std::io::Result<usize>, STACK_FUTURE_SIZE> {
        match self {
            FileReader::Whole(reader) => reader.read_to_end(buf),
            FileReader::Stream { file, len } => {
                buf.reserve(usize::try_from(*len).unwrap_or(usize::MAX));
                StackFuture::from(futures_lite::AsyncReadExt::read_to_end(file, buf))
            }
        }
    }

    fn seekable(&mut self) -> Result<&mut dyn SeekableReader, ReaderNotSeekableError> {
        Ok(self)
    }
}

/// Creates the file at `full_path` and any missing parent directories.
async fn create_for_write(full_path: PathBuf) -> Result<Box<Writer>, AssetWriterError> {
    if let Some(parent) = full_path.parent() {
        async_fs::create_dir_all(parent).await?;
    }
    let file = unblock(move || std::fs::File::create(full_path)).await?;
    let writer: Box<Writer> = Box::new(Unblock::with_capacity(WRITE_PIPE_CAPACITY, file));
    Ok(writer)
}

impl AssetReader for FileAssetReader {
    async fn read<'a>(&'a self, path: &'a Path) -> Result<impl Reader + 'a, AssetReaderError> {
        #[cfg(not(target_os = "windows"))]
        let _guard = maybe_get_semaphore().await;

        let full_path = self.root_path.join(path);
        let open_path = full_path.clone();
        let opened = unblock(move || open_for_read(&open_path))
            .await
            .map_err(|e| map_not_found(e, full_path))?;
        Ok(match opened {
            Opened::Whole(bytes) => FileReader::Whole(VecReader::new(bytes)),
            Opened::Stream { file, len } => FileReader::Stream {
                file: GuardedFile {
                    file: File::from(file),
                    #[cfg(not(target_os = "windows"))]
                    _guard,
                    #[cfg(target_os = "windows")]
                    _lifetime: PhantomData,
                },
                len,
            },
        })
    }

    async fn read_meta<'a>(&'a self, path: &'a Path) -> Result<impl Reader + 'a, AssetReaderError> {
        #[cfg(not(target_os = "windows"))]
        let _guard = maybe_get_semaphore().await;

        let full_path = self.root_path.join(get_meta_path(path));
        let read_path = full_path.clone();
        let bytes = unblock(move || std::fs::read(read_path))
            .await
            .map_err(|e| map_not_found(e, full_path))?;
        Ok(VecReader::new(bytes))
    }

    async fn read_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<Box<PathStream>, AssetReaderError> {
        let full_path = self.root_path.join(path);
        match read_dir(&full_path).await {
            Ok(read_dir) => {
                let root_path = self.root_path.clone();
                let mapped_stream = read_dir.filter_map(move |f| {
                    f.ok().and_then(|dir_entry| {
                        let path = dir_entry.path();
                        // filter out meta files as they are not considered assets
                        if let Some(ext) = path.extension().and_then(|e| e.to_str())
                            && ext.eq_ignore_ascii_case("meta")
                        {
                            return None;
                        }
                        // filter out hidden files. they are not listed by default but are directly targetable
                        if path
                            .file_name()
                            .and_then(|file_name| file_name.to_str())
                            .map(|file_name| file_name.starts_with('.'))
                            .unwrap_or_default()
                        {
                            return None;
                        }
                        let relative_path = path.strip_prefix(&root_path).unwrap();
                        Some(relative_path.to_owned())
                    })
                });
                let read_dir: Box<PathStream> = Box::new(mapped_stream);
                Ok(read_dir)
            }
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Err(AssetReaderError::NotFound(full_path))
                } else {
                    Err(e.into())
                }
            }
        }
    }

    async fn is_directory<'a>(&'a self, path: &'a Path) -> Result<bool, AssetReaderError> {
        let full_path = self.root_path.join(path);
        let metadata = full_path
            .metadata()
            .map_err(|_e| AssetReaderError::NotFound(path.to_owned()))?;
        Ok(metadata.file_type().is_dir())
    }
}

impl AssetWriter for FileAssetWriter {
    async fn write<'a>(&'a self, path: &'a Path) -> Result<Box<Writer>, AssetWriterError> {
        create_for_write(self.root_path.join(path)).await
    }

    async fn write_meta<'a>(&'a self, path: &'a Path) -> Result<Box<Writer>, AssetWriterError> {
        create_for_write(self.root_path.join(get_meta_path(path))).await
    }

    async fn remove<'a>(&'a self, path: &'a Path) -> Result<(), AssetWriterError> {
        let full_path = self.root_path.join(path);
        async_fs::remove_file(full_path).await?;
        Ok(())
    }

    async fn remove_meta<'a>(&'a self, path: &'a Path) -> Result<(), AssetWriterError> {
        let meta_path = get_meta_path(path);
        let full_path = self.root_path.join(meta_path);
        async_fs::remove_file(full_path).await?;
        Ok(())
    }

    async fn rename<'a>(
        &'a self,
        old_path: &'a Path,
        new_path: &'a Path,
    ) -> Result<(), AssetWriterError> {
        let full_old_path = self.root_path.join(old_path);
        let full_new_path = self.root_path.join(new_path);
        if let Some(parent) = full_new_path.parent() {
            async_fs::create_dir_all(parent).await?;
        }
        async_fs::rename(full_old_path, full_new_path).await?;
        Ok(())
    }

    async fn rename_meta<'a>(
        &'a self,
        old_path: &'a Path,
        new_path: &'a Path,
    ) -> Result<(), AssetWriterError> {
        let old_meta_path = get_meta_path(old_path);
        let new_meta_path = get_meta_path(new_path);
        let full_old_path = self.root_path.join(old_meta_path);
        let full_new_path = self.root_path.join(new_meta_path);
        if let Some(parent) = full_new_path.parent() {
            async_fs::create_dir_all(parent).await?;
        }
        async_fs::rename(full_old_path, full_new_path).await?;
        Ok(())
    }

    async fn create_directory<'a>(&'a self, path: &'a Path) -> Result<(), AssetWriterError> {
        let full_path = self.root_path.join(path);
        async_fs::create_dir_all(full_path).await?;
        Ok(())
    }

    async fn remove_directory<'a>(&'a self, path: &'a Path) -> Result<(), AssetWriterError> {
        let full_path = self.root_path.join(path);
        async_fs::remove_dir_all(full_path).await?;
        Ok(())
    }

    async fn remove_empty_directory<'a>(&'a self, path: &'a Path) -> Result<(), AssetWriterError> {
        let full_path = self.root_path.join(path);
        async_fs::remove_dir(full_path).await?;
        Ok(())
    }

    async fn remove_assets_in_directory<'a>(
        &'a self,
        path: &'a Path,
    ) -> Result<(), AssetWriterError> {
        let full_path = self.root_path.join(path);
        async_fs::remove_dir_all(&full_path).await?;
        async_fs::create_dir_all(&full_path).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{format, vec};
    use futures_lite::{future::block_on, AsyncReadExt, AsyncSeekExt, AsyncWriteExt};

    /// A directory under the system temp dir that is removed on drop.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(name: &str) -> Self {
            let path = std::env::temp_dir().join(format!(
                "bevy_asset_file_asset_{name}_{}",
                std::process::id()
            ));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn patterned_bytes(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i % 251) as u8).collect()
    }

    /// Reads `path` through [`Reader::read_to_end`].
    fn read_to_end(reader: &FileAssetReader, path: &Path) -> Vec<u8> {
        block_on(async {
            let mut reader = reader.read(path).await.unwrap();
            let mut bytes = Vec::new();
            Reader::read_to_end(&mut reader, &mut bytes).await.unwrap();
            bytes
        })
    }

    /// Seeks to `offset` in `path`, then reads `len` bytes through [`AsyncRead`].
    fn read_after_seek(reader: &FileAssetReader, path: &Path, offset: u64, len: usize) -> Vec<u8> {
        block_on(async {
            let mut reader = reader.read(path).await.unwrap();
            let reader = reader.seekable().unwrap();
            assert_eq!(reader.seek(SeekFrom::Start(offset)).await.unwrap(), offset);
            let mut bytes = vec![0; len];
            reader.read_exact(&mut bytes).await.unwrap();
            bytes
        })
    }

    #[test]
    fn small_file_reads_whole() {
        let dir = TempDir::new("small");
        let bytes = patterned_bytes(10_000);
        let full_path = dir.0.join("small.bin");
        std::fs::write(&full_path, &bytes).unwrap();
        assert!(matches!(open_for_read(&full_path), Ok(Opened::Whole(_))));

        let reader = FileAssetReader::new(&dir.0);
        let path = Path::new("small.bin");
        assert_eq!(read_to_end(&reader, path), bytes);
        assert_eq!(
            read_after_seek(&reader, path, 1234, 100),
            &bytes[1234..1334]
        );
    }

    #[test]
    fn file_over_limit_streams() {
        let dir = TempDir::new("large");
        let bytes = patterned_bytes(WHOLE_FILE_READ_LIMIT as usize + 1);
        let full_path = dir.0.join("large.bin");
        std::fs::write(&full_path, &bytes).unwrap();
        assert!(matches!(
            open_for_read(&full_path),
            Ok(Opened::Stream { len, .. }) if len == WHOLE_FILE_READ_LIMIT + 1
        ));

        let reader = FileAssetReader::new(&dir.0);
        let path = Path::new("large.bin");
        assert_eq!(read_to_end(&reader, path), bytes);
        let tail = bytes.len() - 16;
        assert_eq!(
            read_after_seek(&reader, path, tail as u64, 16),
            &bytes[tail..]
        );
    }

    #[test]
    fn missing_files_are_not_found() {
        let dir = TempDir::new("missing");
        let reader = FileAssetReader::new(&dir.0);
        let path = Path::new("missing.bin");

        block_on(async {
            assert_eq!(
                reader.read_meta(path).await.err(),
                Some(AssetReaderError::NotFound(dir.0.join("missing.bin.meta")))
            );
            assert_eq!(
                reader.read(path).await.err(),
                Some(AssetReaderError::NotFound(dir.0.join("missing.bin")))
            );
        });
    }

    #[test]
    fn writer_round_trips() {
        let dir = TempDir::new("write");
        let writer = FileAssetWriter::new(&dir.0, false);
        let reader = FileAssetReader::new(&dir.0);
        let path = Path::new("nested/dir/asset.bin");
        let bytes = patterned_bytes(3 * WRITE_PIPE_CAPACITY + 17);
        let meta = b"(meta_format_version: \"1.0\")".to_vec();

        block_on(async {
            let mut asset_writer = writer.write(path).await.unwrap();
            asset_writer.write_all(&bytes).await.unwrap();
            asset_writer.flush().await.unwrap();
            assert_eq!(std::fs::read(dir.0.join(path)).unwrap(), bytes);
            drop(asset_writer);

            let mut meta_writer = writer.write_meta(path).await.unwrap();
            meta_writer.write_all(&meta).await.unwrap();
            meta_writer.close().await.unwrap();

            let mut meta_reader = reader.read_meta(path).await.unwrap();
            let mut read_meta = Vec::new();
            Reader::read_to_end(&mut meta_reader, &mut read_meta)
                .await
                .unwrap();
            assert_eq!(read_meta, meta);
        });
        assert_eq!(read_to_end(&reader, path), bytes);
    }
}
