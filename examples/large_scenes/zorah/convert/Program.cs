using CUE4Parse.FileProvider;
using CUE4Parse.UE4.Assets;
using CUE4Parse.UE4.Assets.Exports;
using CUE4Parse.UE4.Assets.Exports.Actor;
using CUE4Parse.UE4.Assets.Exports.Component;
using CUE4Parse.UE4.Assets.Exports.Component.StaticMesh;
using CUE4Parse.UE4.Assets.Exports.StaticMesh;
using CUE4Parse.UE4.Assets.Exports.Texture;
using CUE4Parse.UE4.Objects.Core.Math;
using CUE4Parse.UE4.Objects.UObject;
using CUE4Parse.UE4.Versions;
using CUE4Parse_Conversion.Textures;
using OodleSharp;
using System.Buffers.Binary;
using System.Collections;
using System.Globalization;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;

return await ZorahConvert.Run(args);

static class ZorahConvert
{
    private static readonly string[] KnownLevels =
    [
        "GreenHouse_Level",
        "Restir_Level",
        "ThroneRoom_Level",
    ];
    // SM_TravelersPalm_A01_01 and SM_Tree_F02_SmallBush_01 are the only meshes
    // that reference these, and Content/Assets/Vegetation/{TravelersPalm_A,
    // Tree_F}/ ship with a Meshes/ subdirectory and nothing else. Ruled out,
    // against the 2026-08-02 download (changelist 275914), by grepping the name
    // table of all 12,750 packages under Content/ for "Tree_F" and
    // "TravelersPalm": the substrings occur only in those two meshes and in the
    // 22 + 2 __ExternalActors__ that place them. So there is no renamed copy in
    // MaterialLibrary/, Materials/ or Merged/, and no ObjectRedirector either -
    // a redirector would still carry the old object name that these meshes ask
    // for. The names below are byte-for-byte the interface paths in each mesh's
    // StaticMaterials array (verify with `inspect SM_Tree_F02_SmallBush_01`),
    // so this is not a spelling or slot-index mismatch.
    private static readonly HashSet<string> KnownMissingProjectMaterials = new(
        [
            "/Game/Assets/Vegetation/TravelersPalm_A/Materials/MI_TravelersPalm_Zorah_A01_Atlas.MI_TravelersPalm_Zorah_A01_Atlas",
            "/Game/Assets/Vegetation/TravelersPalm_A/Materials/MI_TravelersPalm_Zorah_A01_Stem.MI_TravelersPalm_Zorah_A01_Stem",
            "/Game/Assets/Vegetation/Tree_F/Materials/MI_Tree_F01_Bark_pp01.MI_Tree_F01_Bark_pp01",
            "/Game/Assets/Vegetation/Tree_F/Materials/MI_Tree_F01_Leaves_Zorah.MI_Tree_F01_Leaves_Zorah",
        ],
        StringComparer.Ordinal
    );

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
    };
    private const string GeneratedVariableSuffix = "_GEN_VARIABLE";
    private static readonly Dictionary<string, Dictionary<string, UObject>> BlueprintComponentTemplates =
        new(StringComparer.Ordinal);
    private static readonly Dictionary<string, UObject> NoComponentTemplates =
        new(StringComparer.Ordinal);

    public static async Task<int> Run(string[] args)
    {
        if (args.Length < 2)
        {
            Usage();
            return 2;
        }

        var projectRoot = Path.GetFullPath(args[0]);
        var contentRoot = Path.Combine(projectRoot, "Content");
        if (!Directory.Exists(contentRoot))
        {
            Console.Error.WriteLine($"ZORAH_ERROR Content directory does not exist: {contentRoot}");
            return 2;
        }

        Console.WriteLine($"ZORAH_SCAN content={contentRoot}");
        var provider = new DefaultFileProvider(
            contentRoot,
            SearchOption.AllDirectories,
            new VersionContainer(EGame.GAME_UE5_4),
            StringComparer.OrdinalIgnoreCase
        );
        provider.Initialize();
        Console.WriteLine($"ZORAH_SCAN packages={provider.Files.Count}");

        return args[1] switch
        {
            "list" when args.Length == 2 => ListPackages(provider),
            "conversion-api" when args.Length == 2 => DumpConversionApi(),
            "inspect" when args.Length >= 3 => InspectPackages(provider, args[2..]),
            "scene-manifest" when args.Length == 4 => await WriteSceneManifest(
                provider,
                args[2],
                Path.GetFullPath(args[3])
            ),
            "scene-manifests" when args.Length == 3 => await WriteSceneManifests(
                provider,
                Path.GetFullPath(args[2])
            ),
            "material-manifest" when args.Length == 4 => await WriteMaterialManifest(
                provider,
                Path.GetFullPath(args[2]),
                Path.GetFullPath(args[3])
            ),
            "mesh-material-manifest" when args.Length == 4 => await WriteMeshMaterialManifest(
                provider,
                Path.GetFullPath(args[2]),
                Path.GetFullPath(args[3])
            ),
            "texture-export" when args.Length == 4 => await ExportTextures(
                provider,
                Path.GetFullPath(args[2]),
                Path.GetFullPath(args[3])
            ),
            "raw-source" when args.Length == 4 => await WriteRawSource(
                projectRoot,
                args[2],
                Path.GetFullPath(args[3])
            ),
            _ => BadArguments(),
        };
    }

    private static int BadArguments()
    {
        Usage();
        return 2;
    }

    private static void Usage()
    {
        Console.Error.WriteLine(
            "usage:\n" +
            "  ZorahConvert <zorah-project-root> list\n" +
            "  ZorahConvert <zorah-project-root> conversion-api\n" +
            "  ZorahConvert <zorah-project-root> inspect <package-substring>...\n" +
            "  ZorahConvert <zorah-project-root> scene-manifest <level-name> <output.json>\n" +
            "  ZorahConvert <zorah-project-root> scene-manifests <output-directory>\n" +
            "  ZorahConvert <zorah-project-root> material-manifest <input.json> <output.json>\n" +
            "  ZorahConvert <zorah-project-root> mesh-material-manifest <input.json> <output.json>\n" +
            "  ZorahConvert <zorah-project-root> texture-export <material-manifest.json> <output-directory>\n" +
            "  ZorahConvert <zorah-project-root> raw-source <content-relative.uasset> <output.raw>"
        );
    }

    private static async Task<int> WriteMeshMaterialManifest(
        DefaultFileProvider provider,
        string inputPath,
        string outputPath
    )
    {
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"ZORAH_ERROR mesh-material input does not exist: {inputPath}");
            return 2;
        }
        if (File.Exists(outputPath) || Directory.Exists(outputPath))
        {
            Console.Error.WriteLine($"ZORAH_ERROR output already exists: {outputPath}");
            return 2;
        }

        await using var input = File.OpenRead(inputPath);
        var requested = await JsonSerializer.DeserializeAsync<string[]>(input, JsonOptions) ?? [];
        var packageLookup = provider.Files.Keys
            .Where(path => path.EndsWith(".uasset", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(path => path, StringComparer.OrdinalIgnoreCase);
        var meshes = new List<MeshMaterialRecord>();
        var failures = new List<FailureRecord>();

        foreach (var requestedObject in requested.Distinct(StringComparer.Ordinal).Order())
        {
            var objectPath = NormalizeObjectPath(requestedObject);
            var packageKey = ObjectPathToPackageKey(objectPath);
            if (!packageLookup.TryGetValue(packageKey, out var packagePath))
            {
                failures.Add(new FailureRecord(packageKey, "MissingPackage", objectPath));
                continue;
            }
            try
            {
                var mesh = provider.LoadPackage(packagePath).GetExports()
                    .OfType<UStaticMesh>()
                    .FirstOrDefault(candidate => string.Equals(
                        NormalizeObjectPath(ToGameObjectPath(packagePath, candidate.Name)),
                        objectPath,
                        StringComparison.Ordinal
                    )) ?? throw new InvalidDataException(
                        $"package has no matching UStaticMesh export for {objectPath}"
                    );
                var slots = ReadArrayValues(GetTaggedValue(mesh, "StaticMaterials"))
                    .Select((value, index) =>
                    {
                        var fields = ReadStructFields(value);
                        return new MeshMaterialSlotRecord(
                            Index: index,
                            Material: PackageReferencePath(
                                fields.GetValueOrDefault("MaterialInterface")
                            ),
                            SlotName: JsonScalar(
                                fields.GetValueOrDefault("MaterialSlotName")
                            )?.ToString(),
                            ImportedSlotName: JsonScalar(
                                fields.GetValueOrDefault("ImportedMaterialSlotName")
                            )?.ToString()
                        );
                    })
                    .ToArray();
                meshes.Add(new MeshMaterialRecord(
                    objectPath,
                    packagePath,
                    slots,
                    ReadMeshSectionMaterialMap(mesh, "SectionInfoMap")
                ));
            }
            catch (Exception error)
            {
                failures.Add(new FailureRecord(
                    packagePath,
                    error.GetType().FullName ?? error.GetType().Name,
                    error.Message
                ));
            }
        }

        await WriteJsonAtomic(outputPath, new MeshMaterialManifest(
            Format: "zorah-mesh-material-manifest-v2",
            EngineVersion: "5.4",
            Meshes: meshes.OrderBy(mesh => mesh.Object, StringComparer.Ordinal).ToArray(),
            Failures: failures.ToArray()
        ));
        Console.WriteLine(
            $"ZORAH_MESH_MATERIAL_DONE requested={requested.Length} " +
            $"meshes={meshes.Count} failures={failures.Count} output={outputPath}"
        );
        return failures.Count == 0 ? 0 : 1;
    }

    private static async Task<int> WriteRawSource(
        string projectRoot,
        string relativePackage,
        string outputPath
    )
    {
        const uint packageTag = 0x9E2A83C1;
        const uint compressedBufferMagic = 0xB7756362;
        const int compressedHeaderSize = 64;
        const byte oodleMethod = 3;
        var contentRoot = Path.GetFullPath(Path.Combine(projectRoot, "Content"));
        var packagePath = Path.GetFullPath(Path.Combine(contentRoot, relativePackage));
        if (!packagePath.StartsWith(contentRoot + Path.DirectorySeparatorChar, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("raw-source package escapes the Content directory");
        }
        if (File.Exists(outputPath) || Directory.Exists(outputPath))
        {
            throw new IOException($"refusing to overwrite {outputPath}");
        }

        await using var input = File.OpenRead(packagePath);
        using var reader = new BinaryReader(input, System.Text.Encoding.UTF8, leaveOpen: true);
        if (reader.ReadUInt32() != packageTag || input.Length < 20)
        {
            throw new InvalidDataException("not an Unreal package with a trailer");
        }
        input.Seek(-20, SeekOrigin.End);
        _ = reader.ReadUInt64();
        var trailerLength = reader.ReadUInt64();
        if (reader.ReadUInt32() != packageTag || trailerLength == 0 || trailerLength > (ulong)input.Length)
        {
            throw new InvalidDataException("invalid Unreal package trailer footer");
        }
        var trailerOffset = input.Length - checked((long)trailerLength);
        input.Seek(trailerOffset, SeekOrigin.Begin);
        _ = reader.ReadUInt64();
        _ = reader.ReadInt32();
        var headerLength = reader.ReadUInt32();
        var payloadsLength = reader.ReadUInt64();
        var payloadCount = reader.ReadInt32();
        if (payloadCount != 1)
        {
            throw new InvalidDataException($"raw-source package has {payloadCount} trailer payloads");
        }
        var payloadOffset = trailerOffset + headerLength;
        if ((ulong)payloadOffset + payloadsLength > (ulong)input.Length)
        {
            throw new InvalidDataException("trailer payload extends beyond the package");
        }
        input.Seek(payloadOffset, SeekOrigin.Begin);
        var header = reader.ReadBytes(compressedHeaderSize);
        if (header.Length != compressedHeaderSize)
        {
            throw new EndOfStreamException("truncated compressed-buffer header");
        }
        if (BinaryPrimitives.ReadUInt32BigEndian(header) != compressedBufferMagic)
        {
            throw new InvalidDataException("invalid compressed-buffer magic");
        }
        var method = header[8];
        var blockSizeExponent = header[11];
        var blockCount = BinaryPrimitives.ReadUInt32BigEndian(header.AsSpan(12, 4));
        var rawSize = BinaryPrimitives.ReadUInt64BigEndian(header.AsSpan(16, 8));
        var compressedSize = BinaryPrimitives.ReadUInt64BigEndian(header.AsSpan(24, 8));
        if (method != oodleMethod || blockSizeExponent == 0 || blockSizeExponent >= 32)
        {
            throw new InvalidDataException("raw-source needs a supported Oodle compressed buffer");
        }
        var blockSizes = new uint[blockCount];
        for (var index = 0; index < blockSizes.Length; index++)
        {
            var encoded = reader.ReadBytes(4);
            if (encoded.Length != 4)
            {
                throw new EndOfStreamException("truncated compressed-block size table");
            }
            blockSizes[index] = BinaryPrimitives.ReadUInt32BigEndian(encoded);
        }
        var expectedCompressedSize = (ulong)compressedHeaderSize + (ulong)blockCount * 4;
        expectedCompressedSize += blockSizes.Aggregate(0UL, (total, size) => total + size);
        if (expectedCompressedSize != compressedSize)
        {
            throw new InvalidDataException("compressed-buffer size table does not match its header");
        }

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        var temporaryPath = outputPath + $".tmp.{Environment.ProcessId}";
        try
        {
            await using var output = File.Create(temporaryPath);
            var blockSize = 1 << blockSizeExponent;
            ulong written = 0;
            for (var index = 0; index < blockSizes.Length; index++)
            {
                var compressed = reader.ReadBytes(checked((int)blockSizes[index]));
                if (compressed.Length != blockSizes[index])
                {
                    throw new EndOfStreamException($"truncated compressed block {index}");
                }
                var remaining = rawSize - written;
                var rawBlockSize = checked((int)Math.Min((ulong)blockSize, remaining));
                if (compressed.Length >= rawBlockSize)
                {
                    await output.WriteAsync(compressed.AsMemory(0, rawBlockSize));
                }
                else
                {
                    var raw = new byte[rawBlockSize];
                    var decoded = OodleDecompressor.Decompress(compressed, raw);
                    if (decoded != rawBlockSize)
                    {
                        throw new InvalidDataException(
                            $"OodleSharp block {index} decoded {decoded} bytes; expected {rawBlockSize}"
                        );
                    }
                    await output.WriteAsync(raw);
                }
                written += (uint)rawBlockSize;
            }
            if (written != rawSize)
            {
                throw new InvalidDataException($"raw-source wrote {written} bytes; expected {rawSize}");
            }
            await output.FlushAsync();
            File.Move(temporaryPath, outputPath);
        }
        catch
        {
            File.Delete(temporaryPath);
            throw;
        }
        Console.WriteLine(
            $"ZORAH_RAW_SOURCE_DONE package={relativePackage} bytes={rawSize} output={outputPath}"
        );
        return 0;
    }

    private static async Task<int> ExportTextures(
        DefaultFileProvider provider,
        string inputPath,
        string outputDirectory
    )
    {
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"ZORAH_ERROR texture input does not exist: {inputPath}");
            return 2;
        }
        var manifestPath = Path.Combine(outputDirectory, "textures.json");
        if (File.Exists(manifestPath) || Directory.Exists(manifestPath))
        {
            Console.Error.WriteLine($"ZORAH_ERROR output already exists: {manifestPath}");
            return 2;
        }

        await using var input = File.OpenRead(inputPath);
        using var document = await JsonDocument.ParseAsync(input);
        JsonElement referenceArray;
        if (document.RootElement.ValueKind == JsonValueKind.Array)
        {
            referenceArray = document.RootElement;
        }
        else if (!document.RootElement.TryGetProperty("texture_references", out referenceArray))
        {
            Console.Error.WriteLine("ZORAH_ERROR texture input needs an array or texture_references");
            return 2;
        }
        var references = referenceArray.EnumerateArray()
            .Select(element => element.GetString())
            .Where(value => value is not null)
            .Cast<string>()
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Order(StringComparer.Ordinal)
            .ToArray();
        var packageLookup = provider.Files.Keys
            .Where(path => path.EndsWith(".uasset", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(path => path, StringComparer.OrdinalIgnoreCase);
        var platform = Enum.Parse<ETexturePlatform>("DesktopMobile", ignoreCase: true);
        var format = Enum.Parse<ETextureFormat>("PNG", ignoreCase: true);
        var textures = new List<TextureRecord>();
        var failures = new List<FailureRecord>();

        foreach (var reference in references)
        {
            var packageKey = ObjectPathToPackageKey(reference);
            if (!packageLookup.TryGetValue(packageKey, out var packagePath))
            {
                failures.Add(new FailureRecord(packageKey, "MissingPackage", reference));
                continue;
            }
            try
            {
                var objectName = reference[(reference.LastIndexOf('.') + 1)..];
                var texture = provider.LoadPackage(packagePath).GetExports()
                    .OfType<UTexture>()
                    .FirstOrDefault(candidate => candidate.Name == objectName)
                    ?? throw new InvalidDataException("package has no matching texture export");
                var decoded = TextureDecoder.Decode(texture, platform);
                var relativeOutput = Path.ChangeExtension(packagePath, "png")
                    .Replace('\\', '/');
                if (decoded is not null)
                {
                    var encoded = TextureEncoder.Encode(decoded, format, false, out _);
                    var outputPath = Path.Combine(
                        outputDirectory,
                        relativeOutput.Replace('/', Path.DirectorySeparatorChar)
                    );
                    if (File.Exists(outputPath) || Directory.Exists(outputPath))
                    {
                        throw new IOException($"refusing to overwrite {outputPath}");
                    }
                    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
                    var temporaryPath = outputPath + $".tmp.{Environment.ProcessId}";
                    try
                    {
                        await File.WriteAllBytesAsync(temporaryPath, encoded);
                        File.Move(temporaryPath, outputPath);
                    }
                    catch
                    {
                        File.Delete(temporaryPath);
                        throw;
                    }
                    textures.Add(new TextureRecord(
                        Object: reference,
                        Package: packagePath,
                        Output: relativeOutput,
                        Width: decoded.Width,
                        Height: decoded.Height,
                        PixelFormat: decoded.PixelFormat.ToString(),
                        SourceCompression: null,
                        Srgb: texture.GetOrDefault("SRGB", true),
                        IsNormalMap: texture.IsNormalMap,
                        EditorSource: false,
                        PayloadSize: decoded.Data.LongLength,
                        Exported: true,
                        Blocks: []
                    ));
                }
                else
                {
                    var source = ReadStructFields(GetTaggedValue(texture, "Source"));
                    var sourceFormat = JsonScalar(source.GetValueOrDefault("Format"))?.ToString()
                        ?? throw new InvalidDataException("texture Source has no format");
                    var sourceCompression = JsonScalar(
                        source.GetValueOrDefault("CompressionFormat")
                    )?.ToString() ?? "TSCF_None";
                    if (texture is not UTexture2D texture2d)
                    {
                        throw new InvalidDataException(
                            $"editor Source export only supports Texture2D, got {texture.ExportType}"
                        );
                    }
                    var sourceWidth = ToNullableInt(source.GetValueOrDefault("SizeX")) ?? 0;
                    var sourceHeight = ToNullableInt(source.GetValueOrDefault("SizeY")) ?? 0;
                    var payloadSize = texture.EditorData?.PayloadSize ?? 0;
                    var blocks = ReadTextureSourceBlocks(
                        source,
                        sourceWidth,
                        sourceHeight,
                        payloadSize
                    );
                    // Source.SizeX/SizeY describe one tile. ImportedSize describes
                    // the assembled extent for multi-block/UDIM source art. Some
                    // Interchange imports leave ImportedSize at zero, so a single
                    // source block remains the fallback.
                    var width = texture2d.ImportedSize.X;
                    var height = texture2d.ImportedSize.Y;
                    if (width <= 0 || height <= 0)
                    {
                        width = sourceWidth;
                        height = sourceHeight;
                    }
                    if (width <= 0 || height <= 0 || payloadSize <= 0)
                    {
                        throw new InvalidDataException(
                            "texture has neither cooked pixels nor a usable editor Source payload"
                        );
                    }
                    textures.Add(new TextureRecord(
                        Object: reference,
                        Package: packagePath,
                        Output: relativeOutput,
                        Width: width,
                        Height: height,
                        PixelFormat: sourceFormat,
                        SourceCompression: sourceCompression,
                        Srgb: texture.GetOrDefault("SRGB", true),
                        IsNormalMap: texture.IsNormalMap,
                        EditorSource: true,
                        PayloadSize: payloadSize,
                        Exported: false,
                        Blocks: blocks
                    ));
                }
            }
            catch (Exception error)
            {
                failures.Add(new FailureRecord(
                    packagePath,
                    error.GetType().FullName ?? error.GetType().Name,
                    error.Message
                ));
            }
        }

        var manifest = new TextureManifest(
            Format: "zorah-texture-manifest-v1",
            EngineVersion: "5.4",
            Textures: textures.OrderBy(texture => texture.Object, StringComparer.Ordinal).ToArray(),
            Failures: failures.ToArray()
        );
        await WriteJsonAtomic(manifestPath, manifest);
        Console.WriteLine(
            $"ZORAH_TEXTURE_DONE requested={references.Length} textures={textures.Count} " +
            $"failures={failures.Count} output={manifestPath}"
        );
        return failures.Count == 0 ? 0 : 1;
    }

    private static async Task<int> WriteMaterialManifest(
        DefaultFileProvider provider,
        string inputPath,
        string outputPath
    )
    {
        if (!File.Exists(inputPath))
        {
            Console.Error.WriteLine($"ZORAH_ERROR material input does not exist: {inputPath}");
            return 2;
        }
        if (File.Exists(outputPath) || Directory.Exists(outputPath))
        {
            Console.Error.WriteLine($"ZORAH_ERROR output already exists: {outputPath}");
            return 2;
        }

        await using var input = File.OpenRead(inputPath);
        var requested = await JsonSerializer.DeserializeAsync<string[]>(input, JsonOptions) ?? [];
        var packageLookup = provider.Files.Keys
            .Where(path => path.EndsWith(".uasset", StringComparison.OrdinalIgnoreCase))
            .ToDictionary(path => path, StringComparer.OrdinalIgnoreCase);
        var pending = new Queue<string>(requested.Select(NormalizeObjectPath));
        var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var materials = new List<MaterialRecord>();
        var failures = new List<FailureRecord>();

        while (pending.TryDequeue(out var objectPath))
        {
            if (!visited.Add(objectPath))
            {
                continue;
            }
            var packageKey = ObjectPathToPackageKey(objectPath);
            if (!packageLookup.TryGetValue(packageKey, out var packagePath))
            {
                Console.Error.WriteLine(
                    $"ZORAH_MATERIAL_UNRESOLVED object={objectPath} package={packageKey} " +
                    "reason=no-such-package-under-content " +
                    $"known_missing={KnownMissingProjectMaterials.Contains(objectPath)}"
                );
                if (KnownMissingProjectMaterials.Contains(objectPath))
                {
                    // The fixed Zorah 1.1.0 download contains these exact mesh
                    // references but omits their material packages. Preserve
                    // the authored object identity and render an unmistakable
                    // diagnostic material; never substitute a similarly named
                    // package.
                    materials.Add(new MaterialRecord(
                        Package: packageKey,
                        Object: objectPath,
                        Type: "MissingSourceMaterial",
                        Parent: null,
                        Scalars: [new MaterialParameterRecord(
                            "Roughness", "GlobalParameter", -1, 1.0
                        )],
                        Vectors: [new MaterialParameterRecord(
                            "Tint", "GlobalParameter", -1,
                            "FF00FFFF (FLinearColor)"
                        )],
                        Textures: [],
                        StaticSwitches: [],
                        Layers: [],
                        Blends: [],
                        BaseOverrides: []
                    ));
                    continue;
                }
                failures.Add(new FailureRecord(packageKey, "MissingPackage", objectPath));
                continue;
            }

            try
            {
                var exports = provider.LoadPackage(packagePath).GetExports().ToArray();
                var material = exports.FirstOrDefault(candidate =>
                    candidate.Outer is ResolvedPackageObject &&
                    candidate.ExportType.Contains("Material", StringComparison.Ordinal)
                );
                if (material is null)
                {
                    throw new InvalidDataException("package has no top-level material export");
                }

                var parent = PackageReferencePath(GetTaggedValue(material, "Parent"));
                if (parent?.StartsWith("/Game/", StringComparison.Ordinal) == true)
                {
                    pending.Enqueue(parent);
                }
                var layerFunctions = ReadMaterialLayerFunctions(material);
                foreach (var function in layerFunctions.Layers.Concat(layerFunctions.Blends))
                {
                    if (function.StartsWith("/Game/", StringComparison.Ordinal))
                    {
                        pending.Enqueue(function);
                    }
                }
                var expressionDefaults = ReadMaterialExpressionDefaults(material, exports);
                materials.Add(new MaterialRecord(
                    Package: packagePath,
                    Object: ToGameObjectPath(packagePath, material.Name),
                    Type: material.ExportType,
                    Parent: parent,
                    Scalars: MergeMaterialParameters(
                        expressionDefaults.Scalars,
                        ReadMaterialParameters(material, "ScalarParameterValues")
                    ),
                    Vectors: MergeMaterialParameters(
                        expressionDefaults.Vectors,
                        ReadMaterialParameters(material, "VectorParameterValues")
                    ),
                    Textures: MergeMaterialParameters(
                        expressionDefaults.Textures,
                        ReadMaterialParameters(material, "TextureParameterValues")
                    ),
                    StaticSwitches: MergeStaticSwitchParameters(
                        expressionDefaults.StaticSwitches,
                        ReadStaticSwitchParameters(material)
                    ),
                    Layers: layerFunctions.Layers,
                    Blends: layerFunctions.Blends,
                    BaseOverrides: ReadJsonStruct(GetTaggedValue(material, "BasePropertyOverrides"))
                ));
            }
            catch (Exception error)
            {
                failures.Add(new FailureRecord(
                    packagePath,
                    error.GetType().FullName ?? error.GetType().Name,
                    error.Message
                ));
            }
        }

        var manifest = new MaterialManifest(
            Format: "zorah-material-manifest-v2",
            EngineVersion: "5.4",
            Requested: requested.Order(StringComparer.Ordinal).ToArray(),
            Materials: materials.OrderBy(material => material.Object, StringComparer.Ordinal).ToArray(),
            TextureReferences: materials
                .SelectMany(material => material.Textures)
                .Select(parameter => parameter.Value as string)
                .Where(value => value?.StartsWith("/Game/", StringComparison.Ordinal) == true)
                .Cast<string>()
                .Distinct(StringComparer.Ordinal)
                .Order(StringComparer.Ordinal)
                .ToArray(),
            Failures: failures.ToArray()
        );
        await WriteJsonAtomic(outputPath, manifest);
        Console.WriteLine(
            $"ZORAH_MATERIAL_DONE requested={requested.Length} materials={materials.Count} " +
            $"textures={manifest.TextureReferences.Length} failures={failures.Count} output={outputPath}"
        );
        return failures.Count == 0 ? 0 : 1;
    }

    private static string NormalizeObjectPath(string path)
    {
        var normalized = path.Replace('\\', '/');
        if (normalized.StartsWith("Content/", StringComparison.OrdinalIgnoreCase))
        {
            normalized = "/Game/" + normalized["Content/".Length..];
        }
        else if (!normalized.StartsWith("/", StringComparison.Ordinal))
        {
            normalized = "/Game/" + normalized;
        }
        if (normalized.EndsWith(".uasset", StringComparison.OrdinalIgnoreCase))
        {
            normalized = normalized[..^".uasset".Length];
        }
        if (!normalized.Contains('.', StringComparison.Ordinal))
        {
            normalized += "." + Path.GetFileName(normalized);
        }
        return normalized;
    }

    private static string ObjectPathToPackageKey(string objectPath)
    {
        var withoutMount = objectPath.StartsWith("/Game/", StringComparison.OrdinalIgnoreCase)
            ? objectPath["/Game/".Length..]
            : objectPath.TrimStart('/');
        var dot = withoutMount.IndexOf('.');
        if (dot >= 0)
        {
            withoutMount = withoutMount[..dot];
        }
        return withoutMount + ".uasset";
    }

    private static string ToGameObjectPath(string packagePath, string objectName)
    {
        var package = packagePath.EndsWith(".uasset", StringComparison.OrdinalIgnoreCase)
            ? packagePath[..^".uasset".Length]
            : packagePath;
        return $"/Game/{package}.{objectName}";
    }

    private static object? GetTaggedValue(UObject obj, string name) => obj.Properties
        .FirstOrDefault(property => property.Name.Text == name)
        ?.Tag?.GenericValue;

    private static MaterialParameterRecord[] ReadMaterialParameters(UObject material, string name)
    {
        var array = GetTaggedValue(material, name);
        var entries = GetPublicMember(array, "Properties") as IEnumerable;
        if (entries is null)
        {
            return [];
        }

        var result = new List<MaterialParameterRecord>();
        foreach (var entry in entries)
        {
            var fields = ReadStructFields(GetPublicMember(entry, "GenericValue"));
            var info = ReadStructFields(fields.GetValueOrDefault("ParameterInfo"));
            var parameterName = JsonScalar(info.GetValueOrDefault("Name"))?.ToString()
                ?? "unnamed";
            result.Add(new MaterialParameterRecord(
                Name: parameterName,
                Association: JsonScalar(info.GetValueOrDefault("Association"))?.ToString(),
                Index: ToNullableInt(info.GetValueOrDefault("Index")),
                Value: JsonScalar(fields.GetValueOrDefault("ParameterValue"))
            ));
        }
        return result.OrderBy(parameter => parameter.Name, StringComparer.Ordinal).ToArray();
    }

    private static StaticSwitchParameterRecord[] ReadStaticSwitchParameters(UObject material)
    {
        var staticParameters = ReadStructFields(
            GetTaggedValue(material, "StaticParametersRuntime")
        );
        var array = staticParameters.GetValueOrDefault("StaticSwitchParameters");
        var entries = GetPublicMember(array, "Properties") as IEnumerable;
        if (entries is null)
        {
            return [];
        }

        var result = new List<StaticSwitchParameterRecord>();
        foreach (var entry in entries)
        {
            var fields = ReadStructFields(GetPublicMember(entry, "GenericValue"));
            var info = ReadStructFields(fields.GetValueOrDefault("ParameterInfo"));
            result.Add(new StaticSwitchParameterRecord(
                Name: JsonScalar(info.GetValueOrDefault("Name"))?.ToString() ?? "unnamed",
                Association: JsonScalar(info.GetValueOrDefault("Association"))?.ToString(),
                Index: ToNullableInt(info.GetValueOrDefault("Index")),
                Value: ToBool(fields.GetValueOrDefault("Value"), false),
                Override: ToBool(fields.GetValueOrDefault("bOverride"), false)
            ));
        }
        return result
            .OrderBy(parameter => parameter.Name, StringComparer.Ordinal)
            .ThenBy(parameter => parameter.Association, StringComparer.Ordinal)
            .ThenBy(parameter => parameter.Index)
            .ToArray();
    }

    private static MaterialLayerFunctions ReadMaterialLayerFunctions(UObject material)
    {
        var staticParameters = ReadStructFields(
            GetTaggedValue(material, "StaticParametersRuntime")
        );
        var functions = ReadStructFields(staticParameters.GetValueOrDefault("MaterialLayers"));
        return new MaterialLayerFunctions(
            ReadPackageReferenceArray(functions.GetValueOrDefault("Layers")),
            ReadPackageReferenceArray(functions.GetValueOrDefault("Blends"))
        );
    }

    private static string[] ReadPackageReferenceArray(object? value)
    {
        var entries = GetPublicMember(value, "Properties") as IEnumerable;
        if (entries is null)
        {
            return [];
        }
        return entries
            .Cast<object>()
            .Select(entry => PackageReferencePath(GetPublicMember(entry, "GenericValue")))
            .Where(path => path is not null)
            .Cast<string>()
            .ToArray();
    }

    private static MaterialExpressionDefaults ReadMaterialExpressionDefaults(
        UObject material,
        IEnumerable<UObject> exports
    )
    {
        var materialPrefix = material.GetPathName() + ":";
        var scalars = new List<MaterialParameterRecord>();
        var vectors = new List<MaterialParameterRecord>();
        var textures = new List<MaterialParameterRecord>();
        var staticSwitches = new List<StaticSwitchParameterRecord>();
        foreach (var expression in exports.Where(candidate =>
            candidate.GetPathName().StartsWith(materialPrefix, StringComparison.Ordinal) &&
            candidate.ExportType.StartsWith("MaterialExpression", StringComparison.Ordinal)))
        {
            var parameterName = JsonScalar(GetTaggedValue(expression, "ParameterName"))
                ?.ToString();
            if (string.IsNullOrWhiteSpace(parameterName))
            {
                continue;
            }
            var parameter = new MaterialParameterRecord(
                Name: parameterName,
                Association: "GlobalParameter",
                Index: -1,
                Value: null
            );
            if (expression.ExportType.Contains("ScalarParameter", StringComparison.Ordinal))
            {
                scalars.Add(parameter with
                {
                    Value = JsonScalar(GetTaggedValue(expression, "DefaultValue"))
                });
            }
            else if (expression.ExportType.Contains("VectorParameter", StringComparison.Ordinal))
            {
                vectors.Add(parameter with
                {
                    Value = JsonScalar(GetTaggedValue(expression, "DefaultValue"))
                });
            }
            else if (
                expression.ExportType.Contains("Texture", StringComparison.Ordinal) &&
                expression.ExportType.Contains("Parameter", StringComparison.Ordinal)
            )
            {
                textures.Add(parameter with
                {
                    Value = JsonScalar(GetTaggedValue(expression, "Texture"))
                });
            }
            else if (
                expression.ExportType.Contains("StaticBoolParameter", StringComparison.Ordinal) ||
                expression.ExportType.Contains("StaticSwitchParameter", StringComparison.Ordinal)
            )
            {
                staticSwitches.Add(new StaticSwitchParameterRecord(
                    Name: parameterName,
                    Association: "GlobalParameter",
                    Index: -1,
                    Value: ToBool(GetTaggedValue(expression, "DefaultValue"), false),
                    Override: false
                ));
            }
        }
        return new MaterialExpressionDefaults(
            MergeMaterialParameters([], scalars),
            MergeMaterialParameters([], vectors),
            MergeMaterialParameters([], textures),
            MergeStaticSwitchParameters([], staticSwitches)
        );
    }

    private static MaterialParameterRecord[] MergeMaterialParameters(
        IEnumerable<MaterialParameterRecord> defaults,
        IEnumerable<MaterialParameterRecord> overrides
    )
    {
        var merged = new Dictionary<(string Name, string? Association, int? Index), MaterialParameterRecord>();
        foreach (var parameter in defaults.Concat(overrides))
        {
            merged[(parameter.Name, parameter.Association, parameter.Index)] = parameter;
        }
        return merged.Values
            .OrderBy(parameter => parameter.Name, StringComparer.Ordinal)
            .ThenBy(parameter => parameter.Association, StringComparer.Ordinal)
            .ThenBy(parameter => parameter.Index)
            .ToArray();
    }

    private static StaticSwitchParameterRecord[] MergeStaticSwitchParameters(
        IEnumerable<StaticSwitchParameterRecord> defaults,
        IEnumerable<StaticSwitchParameterRecord> overrides
    )
    {
        var merged = new Dictionary<
            (string Name, string? Association, int? Index),
            StaticSwitchParameterRecord
        >();
        foreach (var parameter in defaults.Concat(overrides))
        {
            merged[(parameter.Name, parameter.Association, parameter.Index)] = parameter;
        }
        return merged.Values
            .OrderBy(parameter => parameter.Name, StringComparer.Ordinal)
            .ThenBy(parameter => parameter.Association, StringComparer.Ordinal)
            .ThenBy(parameter => parameter.Index)
            .ToArray();
    }

    private static Dictionary<string, object?> ReadStructFields(object? value)
    {
        var structType = GetPublicMember(value, "StructType");
        if (structType is not null)
        {
            value = structType;
        }
        var properties = GetPublicMember(value, "Properties") as IEnumerable;
        var result = new Dictionary<string, object?>(StringComparer.Ordinal);
        if (properties is null)
        {
            return result;
        }
        foreach (var property in properties)
        {
            var propertyName = GetPublicMember(GetPublicMember(property, "Name"), "Text")
                ?.ToString();
            if (propertyName is null)
            {
                continue;
            }
            var tag = GetPublicMember(property, "Tag");
            result[propertyName] = GetPublicMember(tag, "GenericValue");
        }
        return result;
    }

    private static Dictionary<string, object?> ReadJsonStruct(object? value) =>
        ReadStructFields(value).ToDictionary(
            pair => pair.Key,
            pair => JsonScalar(pair.Value),
            StringComparer.Ordinal
        );

    private static object? JsonScalar(object? value)
    {
        if (value is null || value is string || value is bool || value is byte ||
            value is sbyte || value is short || value is ushort || value is int ||
            value is uint || value is long || value is ulong || value is float ||
            value is double || value is decimal)
        {
            return value;
        }
        if (value is FPackageIndex packageIndex)
        {
            return packageIndex.ResolvedObject?.GetPathName();
        }
        var text = GetPublicMember(value, "Text");
        if (text is string name)
        {
            return name;
        }
        var fields = ReadStructFields(value);
        if (fields.Count != 0)
        {
            return fields.ToDictionary(
                pair => pair.Key,
                pair => JsonScalar(pair.Value),
                StringComparer.Ordinal
            );
        }
        var rgba = new[] { "R", "G", "B", "A" }
            .Select(member => GetPublicMember(value, member))
            .ToArray();
        if (rgba.All(component => component is float or double))
        {
            return rgba;
        }
        return value.GetType().IsEnum ? value.ToString() : value.ToString();
    }

    private static object? GetPublicMember(object? value, string name)
    {
        if (value is null)
        {
            return null;
        }
        var type = value.GetType();
        var property = type.GetProperty(name, BindingFlags.Instance | BindingFlags.Public);
        if (property is not null && property.GetIndexParameters().Length == 0)
        {
            return property.GetValue(value);
        }
        return type.GetField(name, BindingFlags.Instance | BindingFlags.Public)?.GetValue(value);
    }

    private static string? PackageReferencePath(object? value) => value is FPackageIndex index
        ? index.ResolvedObject?.GetPathName()
        : null;

    private static MeshSectionMaterialRecord[] ReadMeshSectionMaterialMap(
        UStaticMesh mesh,
        string propertyName
    )
    {
        var fields = ReadStructFields(GetTaggedValue(mesh, propertyName));
        var map = fields.GetValueOrDefault("Map");
        var entries = GetPublicMember(map, "Properties") as IDictionary;
        if (entries is null)
        {
            return [];
        }
        var result = new List<MeshSectionMaterialRecord>();
        foreach (DictionaryEntry entry in entries)
        {
            var keyValue = GetPublicMember(entry.Key, "GenericValue") ?? entry.Key;
            var packed = keyValue switch
            {
                byte value => (uint)value,
                ushort value => value,
                uint value => value,
                int value when value >= 0 => (uint)value,
                _ => throw new InvalidDataException(
                    $"{propertyName} has unsupported key type {keyValue?.GetType().FullName ?? "null"}"
                ),
            };
            var rawValue = GetPublicMember(entry.Value, "GenericValue") ?? entry.Value;
            var section = ReadStructFields(rawValue);
            var materialIndex = ToNullableInt(section.GetValueOrDefault("MaterialIndex"))
                ?? throw new InvalidDataException(
                    $"{propertyName}[{packed}] has no integer MaterialIndex"
                );
            result.Add(new MeshSectionMaterialRecord(
                Lod: checked((int)(packed >> 16)),
                Section: checked((int)(packed & 0xffff)),
                MaterialIndex: materialIndex
            ));
        }
        return result
            .OrderBy(record => record.Lod)
            .ThenBy(record => record.Section)
            .ToArray();
    }

    private static int? ToNullableInt(object? value) => value switch
    {
        byte typed => typed,
        sbyte typed => typed,
        short typed => typed,
        ushort typed => typed,
        int typed => typed,
        uint typed when typed <= int.MaxValue => (int)typed,
        _ => null,
    };

    private static long? ToNullableLong(object? value) => value switch
    {
        byte typed => typed,
        sbyte typed => typed,
        short typed => typed,
        ushort typed => typed,
        int typed => typed,
        uint typed => typed,
        long typed => typed,
        ulong typed when typed <= long.MaxValue => (long) typed,
        _ => null,
    };

    private static object?[] ReadArrayValues(object? value)
    {
        var entries = GetPublicMember(value, "Properties") as IEnumerable;
        return entries is null
            ? []
            : entries.Cast<object>()
                .Select(entry => GetPublicMember(entry, "GenericValue"))
                .ToArray();
    }

    private static TextureBlockRecord[] ReadTextureSourceBlocks(
        Dictionary<string, object?> source,
        int sourceWidth,
        int sourceHeight,
        long payloadSize
    )
    {
        var blockFields = ReadArrayValues(source.GetValueOrDefault("Blocks"))
            .Select(ReadStructFields)
            .ToList();
        blockFields.Insert(0, new Dictionary<string, object?>(StringComparer.Ordinal)
        {
            // FTextureSource keeps the first UDIM tile out of Blocks and stores
            // its coordinates in BaseBlockX/BaseBlockY.
            ["BlockX"] = ToNullableInt(source.GetValueOrDefault("BaseBlockX")) ?? 0,
            ["BlockY"] = ToNullableInt(source.GetValueOrDefault("BaseBlockY")) ?? 0,
            ["SizeX"] = sourceWidth,
            ["SizeY"] = sourceHeight,
        });
        var offsets = ReadArrayValues(source.GetValueOrDefault("BlockDataOffsets"))
            .Select(ToNullableLong)
            .Where(value => value is not null)
            .Select(value => value!.Value)
            .ToArray();
        if (offsets.Length == 0 && blockFields.Count == 1)
        {
            offsets = [0];
        }
        if (offsets.Length != blockFields.Count)
        {
            throw new InvalidDataException(
                $"texture Source has {blockFields.Count} blocks but {offsets.Length} offsets"
            );
        }
        var sortedOffsets = offsets.Distinct().Order().ToArray();
        if (sortedOffsets.Length != offsets.Length)
        {
            throw new InvalidDataException("texture Source has duplicate block offsets");
        }
        var result = new TextureBlockRecord[blockFields.Count];
        for (var index = 0; index < result.Length; index++)
        {
            var fields = blockFields[index];
            var offset = offsets[index];
            var sortedIndex = Array.BinarySearch(sortedOffsets, offset);
            var end = sortedIndex + 1 < sortedOffsets.Length
                ? sortedOffsets[sortedIndex + 1]
                : payloadSize;
            if (offset < 0 || end <= offset || end > payloadSize)
            {
                throw new InvalidDataException("texture Source has invalid block offsets");
            }
            result[index] = new TextureBlockRecord(
                BlockX: ToNullableInt(fields.GetValueOrDefault("BlockX")) ?? 0,
                BlockY: ToNullableInt(fields.GetValueOrDefault("BlockY")) ?? 0,
                Width: ToNullableInt(fields.GetValueOrDefault("SizeX")) ?? sourceWidth,
                Height: ToNullableInt(fields.GetValueOrDefault("SizeY")) ?? sourceHeight,
                PayloadOffset: offset,
                PayloadSize: end - offset
            );
        }
        if (result.DistinctBy(block => (block.BlockX, block.BlockY)).Count() != result.Length)
        {
            throw new InvalidDataException("texture Source has duplicate block coordinates");
        }
        return result;
    }

    private static int ListPackages(DefaultFileProvider provider)
    {
        foreach (var path in provider.Files.Keys.Order().Take(20))
        {
            Console.WriteLine($"ZORAH_PACKAGE {path}");
        }
        return 0;
    }

    private static int DumpConversionApi()
    {
        var assembly = Assembly.Load("CUE4Parse-Conversion");
        foreach (var type in assembly.GetTypes()
            .Where(type => type.Namespace?.Contains("Textures", StringComparison.Ordinal) == true)
            .OrderBy(type => type.FullName, StringComparer.Ordinal))
        {
            Console.WriteLine($"ZORAH_API_TYPE {type.FullName}");
            foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance)
                .Where(method => method.DeclaringType == type)
                .OrderBy(method => method.Name, StringComparer.Ordinal))
            {
                var parameters = string.Join(", ", method.GetParameters().Select(parameter =>
                    $"{parameter.ParameterType.FullName} {parameter.Name}"));
                Console.WriteLine(
                    $"ZORAH_API_METHOD static={method.IsStatic} return={method.ReturnType.FullName} " +
                    $"name={method.Name} params=({parameters})"
                );
            }
        }
        return 0;
    }

    private static int InspectPackage(DefaultFileProvider provider, string queryArgument)
    {
        var query = queryArgument.Replace('\\', '/');
        var candidates = provider.Files.Keys
            .Where(path => path.Contains(query, StringComparison.OrdinalIgnoreCase))
            .Order()
            .ToArray();
        Console.WriteLine($"ZORAH_MATCH query={query} count={candidates.Length}");

        foreach (var path in candidates.Take(20))
        {
            Console.WriteLine($"ZORAH_MATCHED_PACKAGE {path}");
        }

        if (candidates.Length != 1)
        {
            return candidates.Length == 0 ? 1 : 2;
        }

        var package = provider.LoadPackage(candidates[0]);
        var objects = package.GetExports().ToArray();
        Console.WriteLine($"ZORAH_LOAD package={candidates[0]} objects={objects.Length}");
        foreach (var obj in objects)
        {
            Console.WriteLine(
                $"ZORAH_OBJECT class={obj.ExportType} runtime_type={obj.GetType().FullName} " +
                $"name={obj.Name} path={obj.GetPathName()} " +
                $"class_path={obj.Class?.GetPathName() ?? "None"} " +
                $"template={obj.Template?.GetPathName() ?? "None"}"
            );
            if (IsExternalActorRoot(obj))
            {
                var rootComponent = FindRootComponent(obj, objects);
                var rootTransform = ConvertObjectTransform(rootComponent);
                Console.WriteLine(
                    $"ZORAH_ACTOR_ROOT component={rootComponent?.Name ?? "None"} " +
                    $"type={rootComponent?.ExportType ?? "None"} " +
                    $"translation={Format(rootTransform.Translation)} " +
                    $"rotation={Format(rootTransform.Rotation)} " +
                    $"scale={Format(rootTransform.Scale)}"
                );
            }
            if (obj is UStaticMesh mesh)
            {
                foreach (var (material, index) in ReadArrayValues(
                    GetTaggedValue(mesh, "StaticMaterials")
                ).Select((value, index) => (ReadStructFields(value), index)))
                {
                    Console.WriteLine(
                        $"ZORAH_STATIC_MATERIAL_EXACT index={index} " +
                        $"interface={PackageReferencePath(material.GetValueOrDefault("MaterialInterface")) ?? "None"} " +
                        $"slot={JsonScalar(material.GetValueOrDefault("MaterialSlotName")) ?? "None"} " +
                        $"imported_slot={JsonScalar(material.GetValueOrDefault("ImportedMaterialSlotName")) ?? "None"}"
                    );
                }
                Console.WriteLine(
                    $"ZORAH_STATIC_MESH render_data={(mesh.RenderData is null ? "missing" : "present")} " +
                    $"nanite={(mesh.RenderData?.NaniteResources is null ? "missing" : "present")}"
                );
                DumpShape(mesh);
                foreach (var material in mesh.StaticMaterials ?? [])
                {
                    Console.WriteLine("ZORAH_STATIC_MATERIAL");
                    DumpMembers(material);
                }
            }
            if (obj is UTexture texture)
            {
                Console.WriteLine("ZORAH_TEXTURE_EDITOR_DATA");
                if (texture.EditorData is not null)
                {
                    DumpMembers(texture.EditorData);
                    Console.WriteLine("ZORAH_TEXTURE_COMPRESSED_PAYLOAD");
                    DumpMembers(texture.EditorData.Payload);
                    foreach (var method in texture.EditorData.Payload.GetType()
                        .GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.Static)
                        .Where(method => method.DeclaringType == texture.EditorData.Payload.GetType()))
                    {
                        Console.WriteLine(
                            $"ZORAH_TEXTURE_PAYLOAD_METHOD return={method.ReturnType.FullName} " +
                            $"name={method.Name} params=({string.Join(",", method.GetParameters().Select(parameter => parameter.ParameterType.FullName))})"
                        );
                    }
                    var payloadType = texture.EditorData.Payload.GetType();
                    foreach (var method in payloadType.Assembly.GetTypes()
                        .SelectMany(type => type.GetMethods(
                            BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static
                        ))
                        .Where(method =>
                            method.Name.Contains("Decompress", StringComparison.OrdinalIgnoreCase) ||
                            method.GetParameters().Any(parameter =>
                                parameter.ParameterType == payloadType ||
                                parameter.ParameterType.GetElementType() == payloadType
                            )
                        ))
                    {
                        Console.WriteLine(
                            $"ZORAH_TEXTURE_PAYLOAD_EXTENSION declaring={method.DeclaringType?.FullName} " +
                            $"public={method.IsPublic} return={method.ReturnType.FullName} " +
                            $"name={method.Name} params=({string.Join(",", method.GetParameters().Select(parameter => parameter.ParameterType.FullName))})"
                        );
                    }
                }
                foreach (var pair in ReadStructFields(GetTaggedValue(texture, "Source")))
                {
                    Console.WriteLine(
                        $"ZORAH_TEXTURE_SOURCE name={pair.Key} " +
                        $"type={pair.Value?.GetType().FullName ?? "null"} value={Describe(pair.Value)}"
                    );
                    if (GetPublicMember(pair.Value, "Properties") is not null)
                    {
                        DumpMembers(pair.Value!);
                        var entries = GetPublicMember(pair.Value, "Properties") as IEnumerable;
                        var entryIndex = 0;
                        foreach (var entry in entries ?? Array.Empty<object>())
                        {
                            var generic = GetPublicMember(entry, "GenericValue");
                            Console.WriteLine(
                                $"ZORAH_TEXTURE_SOURCE_ENTRY name={pair.Key} index={entryIndex} " +
                                $"value={JsonSerializer.Serialize(JsonScalar(generic), JsonOptions)}"
                            );
                            entryIndex++;
                        }
                    }
                }
            }
            if (obj is UStaticMeshComponent component)
            {
                var record = ConvertComponent(
                    provider,
                    component,
                    InspectArchetypes(provider, component, objects)
                );
                Console.WriteLine(
                    $"ZORAH_STATIC_MESH_COMPONENT mesh={record.Mesh ?? "None"} " +
                    $"translation={Format(record.Transform.Translation)} " +
                    $"rotation={Format(record.Transform.Rotation)} " +
                    $"scale={Format(record.Transform.Scale)} " +
                    $"instances={record.Instances?.Length ?? 0}"
                );
                if (record.Mesh is null)
                {
                    DumpShape(component);
                    foreach (var property in component.Properties)
                    {
                        Console.WriteLine("ZORAH_PROPERTY_TAG");
                        DumpMembers(property);
                        if (property.Tag is not null)
                        {
                            DumpMembers(property.Tag);
                        }
                    }
                }
            }
            if (obj is USceneComponent sceneComponent)
            {
                var transform = ConvertTransform(sceneComponent.GetRelativeTransform());
                Console.WriteLine(
                    $"ZORAH_SCENE_COMPONENT translation={Format(transform.Translation)} " +
                    $"rotation={Format(transform.Rotation)} scale={Format(transform.Scale)}"
                );
            }
            if (obj.ExportType.Contains("LightComponent", StringComparison.Ordinal))
            {
                Console.WriteLine("ZORAH_LIGHT_COMPONENT");
                Console.WriteLine(
                    "ZORAH_LIGHT_RECORD " +
                    JsonSerializer.Serialize(
                        ConvertLightComponent(obj, InspectArchetypes(provider, obj, objects)),
                        JsonOptions
                    )
                );
                DumpShape(obj);
                foreach (var property in obj.Properties)
                {
                    Console.WriteLine(
                        $"ZORAH_LIGHT_PROPERTY name={property.Name.Text} " +
                        $"value={Describe(property.Tag?.GenericValue)}"
                    );
                    if (property.Tag?.GenericValue is not null)
                    {
                        DumpNested(
                            $"light.{property.Name.Text}",
                            property.Tag.GenericValue,
                            0,
                            new HashSet<object>(ReferenceEqualityComparer.Instance)
                        );
                    }
                }
            }
            if (obj.ExportType == "SkyAtmosphereComponent")
            {
                Console.WriteLine(
                    "ZORAH_SKY_ATMOSPHERE_RECORD " +
                    JsonSerializer.Serialize(ConvertSkyAtmosphereComponent(obj), JsonOptions)
                );
                DumpShape(obj);
            }
            if (obj.ExportType == "ExponentialHeightFogComponent")
            {
                Console.WriteLine(
                    "ZORAH_HEIGHT_FOG_RECORD " +
                    JsonSerializer.Serialize(ConvertHeightFogComponent(obj), JsonOptions)
                );
                DumpShape(obj);
            }
            if (obj.ExportType == "PostProcessVolume")
            {
                Console.WriteLine("ZORAH_POST_PROCESS_VOLUME");
                Console.WriteLine(
                    "ZORAH_POST_PROCESS_SETTINGS " +
                    JsonSerializer.Serialize(
                        ReadJsonStruct(GetTaggedValue(obj, "Settings")),
                        JsonOptions
                    )
                );
                foreach (var property in obj.Properties)
                {
                    Console.WriteLine(
                        $"ZORAH_POST_PROCESS_PROPERTY name={property.Name.Text} " +
                        $"value={Describe(property.Tag?.GenericValue)}"
                    );
                }
            }
            if (obj.ExportType.Contains("MeshDescription", StringComparison.Ordinal))
            {
                DumpShape(obj);
            }
            if (obj.ExportType.Contains("Material", StringComparison.Ordinal) ||
                obj.ExportType.Contains("Texture", StringComparison.Ordinal))
            {
                if (obj.ExportType.Contains("Material", StringComparison.Ordinal))
                {
                    Console.WriteLine(
                        "ZORAH_STATIC_SWITCHES " +
                        JsonSerializer.Serialize(ReadStaticSwitchParameters(obj), JsonOptions)
                    );
                    var staticParameters = ReadStructFields(
                        GetTaggedValue(obj, "StaticParametersRuntime")
                    );
                    if (staticParameters.TryGetValue("MaterialLayers", out var materialLayers))
                    {
                        DumpNested(
                            "material_layers",
                            materialLayers,
                            -3,
                            new HashSet<object>(ReferenceEqualityComparer.Instance)
                        );
                    }
                }
                DumpShape(obj);
                foreach (var property in obj.Properties)
                {
                    Console.WriteLine("ZORAH_ASSET_PROPERTY_TAG");
                    DumpMembers(property);
                    if (property.Tag is not null)
                    {
                        DumpMembers(property.Tag);
                        if (property.Name.Text is
                            "Parent" or
                            "ScalarParameterValues" or
                            "VectorParameterValues" or
                            "TextureParameterValues" or
                            "StaticParametersRuntime" or
                            "StaticParameters" or
                            "BasePropertyOverrides")
                        {
                            DumpNested(
                                $"property.{property.Name.Text}",
                                property.Tag.GenericValue,
                                0,
                                new HashSet<object>(ReferenceEqualityComparer.Instance)
                            );
                        }
                    }
                }
            }
        }
        return 0;
    }

    private static ComponentArchetypes InspectArchetypes(
        DefaultFileProvider provider,
        UObject component,
        UObject[] packageObjects
    )
    {
        var actor = FindOwningActor(component);
        return new ComponentArchetypes(
            actor,
            actor is null ? null : FindRootComponent(actor, packageObjects),
            actor is null
                ? Array.Empty<UObject>()
                : packageObjects.Where(obj => FindOwningActor(obj)?.Name == actor.Name),
            LoadBlueprintComponentTemplates(provider, actor?.Class?.GetPathName())
        );
    }

    private static int InspectPackages(DefaultFileProvider provider, string[] queries)
    {
        foreach (var query in queries)
        {
            var result = InspectPackage(provider, query);
            if (result != 0)
            {
                return result;
            }
        }
        return 0;
    }

    private static async Task<int> WriteSceneManifest(
        DefaultFileProvider provider,
        string level,
        string outputPath
    )
    {
        if (!KnownLevels.Contains(level, StringComparer.Ordinal))
        {
            Console.Error.WriteLine(
                $"ZORAH_ERROR unknown level {level}; expected one of {string.Join(", ", KnownLevels)}"
            );
            return 2;
        }
        if (File.Exists(outputPath) || Directory.Exists(outputPath))
        {
            Console.Error.WriteLine($"ZORAH_ERROR output already exists: {outputPath}");
            return 2;
        }

        var prefix = $"__ExternalActors__/Levels/{level}/";
        var packagePaths = provider.Files.Keys
            .Where(path => path.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            .Where(path => path.EndsWith(".uasset", StringComparison.OrdinalIgnoreCase))
            .Order(StringComparer.Ordinal)
            .ToArray();
        Console.WriteLine($"ZORAH_LEVEL level={level} actor_packages={packagePaths.Length}");

        var actors = new List<ActorRecord>();
        var failures = new List<FailureRecord>();
        var actorTypes = new Dictionary<string, int>(StringComparer.Ordinal);
        var componentTypes = new Dictionary<string, int>(StringComparer.Ordinal);
        var referencedMeshes = new HashSet<string>(StringComparer.Ordinal);
        var packagesWithoutActors = 0;

        for (var packageIndex = 0; packageIndex < packagePaths.Length; packageIndex++)
        {
            var packagePath = packagePaths[packageIndex];
            try
            {
                var objects = provider.LoadPackage(packagePath).GetExports().ToArray();
                var packageActors = objects
                    .Where(IsExternalActorRoot)
                    .OrderBy(actor => actor.Name)
                    .ToArray();
                if (packageActors.Length == 0)
                {
                    packagesWithoutActors++;
                }

                foreach (var actor in packageActors)
                {
                    Increment(actorTypes, actor.ExportType);
                    var rootComponent = FindRootComponent(actor, objects);
                    var owned = objects
                        .Where(component => FindOwningActor(component)?.Name == actor.Name)
                        .ToArray();
                    var archetypes = new ComponentArchetypes(
                        actor,
                        rootComponent,
                        owned,
                        LoadBlueprintComponentTemplates(provider, actor.Class?.GetPathName())
                    );
                    var components = owned
                        .OfType<UStaticMeshComponent>()
                        .OrderBy(component => component.Name)
                        .Select(component => ConvertComponent(provider, component, archetypes))
                        .ToArray();

                    var lights = owned
                        .Where(component => component.ExportType.EndsWith(
                            "LightComponent",
                            StringComparison.Ordinal
                        ))
                        .OrderBy(component => component.Name)
                        .Select(component => ConvertLightComponent(component, archetypes))
                        .Concat(ConvertKnownBlueprintLights(provider, actor))
                        .ToArray();

                    var atmosphere = owned
                        .Where(component => component.ExportType == "SkyAtmosphereComponent")
                        .OrderBy(component => component.Name)
                        .Select(ConvertSkyAtmosphereComponent)
                        .FirstOrDefault();
                    var heightFog = owned
                        .Where(component => component.ExportType == "ExponentialHeightFogComponent")
                        .OrderBy(component => component.Name)
                        .Select(ConvertHeightFogComponent)
                        .FirstOrDefault();

                    foreach (var component in components)
                    {
                        Increment(componentTypes, component.Type);
                        if (component.Mesh is not null)
                        {
                            referencedMeshes.Add(component.Mesh);
                        }
                    }

                    actors.Add(new ActorRecord(
                        Package: packagePath,
                        Name: actor.Name,
                        Label: actor.GetOrDefault<string?>("ActorLabel", null),
                        Type: actor.ExportType,
                        Class: actor.Class?.GetPathName(),
                        Transform: ConvertObjectTransform(rootComponent),
                        Hidden: actor.GetOrDefault("bHidden", false),
                        Components: components,
                        Lights: lights,
                        Atmosphere: atmosphere,
                        HeightFog: heightFog,
                        PostProcess: actor.ExportType == "PostProcessVolume"
                            ? ConvertPostProcessVolume(actor)
                            : null
                    ));
                }
            }
            catch (Exception error)
            {
                failures.Add(new FailureRecord(
                    Package: packagePath,
                    ErrorType: error.GetType().FullName ?? error.GetType().Name,
                    Message: error.Message
                ));
            }

            if ((packageIndex + 1) % 250 == 0 || packageIndex + 1 == packagePaths.Length)
            {
                Console.WriteLine(
                    $"ZORAH_LEVEL_PROGRESS level={level} loaded={packageIndex + 1}/{packagePaths.Length} " +
                    $"actors={actors.Count} failures={failures.Count}"
                );
            }
        }

        var manifest = new SceneManifest(
            Format: "zorah-scene-manifest-v3",
            EngineVersion: "5.4",
            Level: level,
            SourceMap: $"Levels/{level}.umap",
            ExternalActorPrefix: prefix,
            ActorPackageCount: packagePaths.Length,
            PackagesWithoutActors: packagesWithoutActors,
            ActorTypeCounts: actorTypes.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(),
            StaticMeshComponentTypeCounts: componentTypes
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(),
            UnresolvedStaticMeshComponents: actors
                .SelectMany(actor => actor.Components)
                .Count(component => component.Mesh is null),
            ReferencedMeshes: referencedMeshes.Order(StringComparer.Ordinal).ToArray(),
            Actors: actors.OrderBy(actor => actor.Package, StringComparer.Ordinal)
                .ThenBy(actor => actor.Name, StringComparer.Ordinal)
                .ToArray(),
            Failures: failures.ToArray()
        );

        var outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }
        var temporaryPath = outputPath + $".tmp.{Environment.ProcessId}";
        try
        {
            await using (var output = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None
            ))
            {
                await JsonSerializer.SerializeAsync(output, manifest, JsonOptions);
                await output.WriteAsync("\n"u8.ToArray());
            }
            File.Move(temporaryPath, outputPath);
        }
        catch
        {
            File.Delete(temporaryPath);
            throw;
        }

        Console.WriteLine(
            $"ZORAH_LEVEL_DONE level={level} actors={manifest.Actors.Length} " +
            $"lights={manifest.Actors.Sum(actor => actor.Lights.Length)} " +
            $"meshes={manifest.ReferencedMeshes.Length} failures={manifest.Failures.Length} " +
            $"unresolved_mesh_components={manifest.UnresolvedStaticMeshComponents} " +
            $"output={outputPath}"
        );
        return failures.Count == 0 ? 0 : 1;
    }

    private static async Task<int> WriteSceneManifests(
        DefaultFileProvider provider,
        string outputDirectory
    )
    {
        Directory.CreateDirectory(outputDirectory);
        foreach (var level in KnownLevels)
        {
            var result = await WriteSceneManifest(
                provider,
                level,
                Path.Combine(outputDirectory, level + ".json")
            );
            if (result != 0)
            {
                return result;
            }
        }
        return 0;
    }

    private static async Task WriteJsonAtomic<T>(string outputPath, T value)
    {
        var outputDirectory = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(outputDirectory))
        {
            Directory.CreateDirectory(outputDirectory);
        }
        var temporaryPath = outputPath + $".tmp.{Environment.ProcessId}";
        try
        {
            await using (var output = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None
            ))
            {
                await JsonSerializer.SerializeAsync(output, value, JsonOptions);
                await output.WriteAsync("\n"u8.ToArray());
            }
            File.Move(temporaryPath, outputPath);
        }
        catch
        {
            File.Delete(temporaryPath);
            throw;
        }
    }

    private static bool IsExternalActorRoot(UObject obj) =>
        obj.Outer is ResolvedPackageObject &&
        obj.ExportType is not "MetaData" &&
        obj.Name is not "PackageMetaData";

    private static UObject? FindOwningActor(UObject obj)
    {
        var outer = obj.Outer;
        while (outer is not null)
        {
            var loaded = outer.Load();
            if (loaded is not null && IsExternalActorRoot(loaded))
            {
                return loaded;
            }
            outer = loaded?.Outer;
        }
        return null;
    }

    private static UObject? FindRootComponent(UObject actor, UObject[] packageObjects)
    {
        try
        {
            var root = actor.GetOrDefault<FPackageIndex?>("RootComponent")?.Load();
            if (root is not null)
            {
                return root;
            }
        }
        catch
        {
            // Blueprint actor instances occasionally leave RootComponent to their
            // stripped editor template. Fall back to the exported component name.
        }

        var owned = packageObjects
            .Where(obj => FindOwningActor(obj)?.Name == actor.Name)
            .ToArray();
        return owned.FirstOrDefault(obj => obj.Name is "DefaultSceneRoot" or "DefaultSceneRoot_0")
            ?? owned.FirstOrDefault(obj => obj.ExportType.EndsWith(
                "RootComponent",
                StringComparison.Ordinal
            ))
            ?? (owned.Length == 1 ? owned[0] : null);
    }

    private static StaticMeshComponentRecord ConvertComponent(
        DefaultFileProvider provider,
        UStaticMeshComponent component,
        ComponentArchetypes archetypes
    )
    {
        var chain = archetypes.Chain(component);
        var meshPath = PackageReferencePath(GetTaggedValue(component, "StaticMesh"))
            ?? component.GetStaticMesh().ResolvedObject?.GetPathName();
        var meshResolution = meshPath is null ? null : "exact-component-property";
        if (meshPath is null && chain.Length > 1)
        {
            var template = chain[1];
            meshPath = PackageReferencePath(GetTaggedValue(template, "StaticMesh"))
                ?? (template as UStaticMeshComponent)?.GetStaticMesh().ResolvedObject
                    ?.GetPathName();
            meshResolution = meshPath is null ? null : "exact-blueprint-component-template";
        }
        // OverrideMaterials is indexed by static-mesh material slot.
        // Preserve explicit null entries so later overrides do not shift.
        var overrides = ReadArrayValues(ReadTaggedValue(chain, "OverrideMaterials"))
            .Select(entry =>
                (entry as FPackageIndex)?.ResolvedObject?.GetPathName() ?? string.Empty)
            .ToArray();

        TransformRecord[]? instances = null;
        if (component is UInstancedStaticMeshComponent instanced)
        {
            instances = instanced.GetInstances()
                .Select(instance => ConvertTransform(instance.TransformData))
                .ToArray();
        }

        return new StaticMeshComponentRecord(
            Name: component.Name,
            Type: component.ExportType,
            Template: component.Template?.GetPathName(),
            Mesh: meshPath,
            MeshResolution: meshResolution,
            Transform: archetypes.TransformRelativeToActor(component),
            Visible: ReadBool(chain, "bVisible", true),
            HiddenInGame: ReadBool(chain, "bHiddenInGame", false),
            CastShadow: ReadBool(chain, "CastShadow", true),
            OverrideMaterials: overrides,
            Instances: instances,
            MissingReason: meshPath is null
                ? ClassifyMissingMesh(
                    provider,
                    component,
                    archetypes.ActorType,
                    archetypes.ActorClass
                )
                : null
        );
    }

    private static string ClassifyMissingMesh(
        DefaultFileProvider provider,
        UStaticMeshComponent component,
        string? actorType,
        string? actorClassPath
    )
    {
        var actorDirectory = ActorPackageDirectory(actorClassPath);
        var stems = new[] { component.Name };
        var relatedPackageExists = provider.Files.Keys
            .Where(path => path.EndsWith(".uasset", StringComparison.OrdinalIgnoreCase))
            .Where(path => actorDirectory is null ||
                path.StartsWith(actorDirectory, StringComparison.OrdinalIgnoreCase))
            .Any(path => stems.Contains(Path.GetFileNameWithoutExtension(path)));
        if (!relatedPackageExists)
        {
            return "source-package-not-in-sample";
        }
        return actorType == "StaticMeshActor"
            ? "stripped-static-mesh-reference"
            : "unresolved-blueprint-template-mesh";
    }

    private static TransformRecord ConvertObjectTransform(UObject? component)
    {
        if (component is USceneComponent sceneComponent)
        {
            return ConvertTransform(sceneComponent.GetRelativeTransform());
        }
        if (component is null)
        {
            return ConvertTransform(FTransform.Identity);
        }

        var location = component.GetOrDefault("RelativeLocation", FVector.ZeroVector);
        var rotation = component.GetOrDefault("RelativeRotation", FRotator.ZeroRotator);
        var scale = component.GetOrDefault("RelativeScale3D", FVector.OneVector);
        return ConvertTransform(new FTransform(rotation, location, scale));
    }

    private static LightComponentRecord ConvertLightComponent(
        UObject component,
        ComponentArchetypes archetypes
    )
    {
        var chain = archetypes.Chain(component);
        var type = component.ExportType switch
        {
            "PointLightComponent" => "point",
            "SpotLightComponent" => "spot",
            "DirectionalLightComponent" => "directional",
            "SkyLightComponent" => "sky",
            _ => component.ExportType,
        };
        // UE class defaults for the delta-elided properties: ULocalLightComponent
        // ships 5000 unitless, UDirectionalLightComponent 10 lux, and
        // USkyLightComponent 1. A light that authored candelas or lumens
        // serializes IntensityUnits, so an absent tag means unitless.
        var defaultIntensity = type switch
        {
            "directional" => 10.0,
            "sky" => 1.0,
            _ => 5000.0,
        };
        var defaultUnits = type == "directional" ? "Lux" : "Unitless";
        return new LightComponentRecord(
            Name: component.Name,
            Type: type,
            ComponentType: component.ExportType,
            Transform: archetypes.TransformRelativeToActor(component),
            Visible: ReadBool(chain, "bVisible", true),
            HiddenInGame: ReadBool(chain, "bHiddenInGame", false),
            AffectsWorld: ReadBool(chain, "bAffectsWorld", true),
            CastShadows: ReadBool(chain, "CastShadows", true),
            Intensity: ReadDouble(chain, "Intensity", defaultIntensity),
            IntensityUnits: ReadName(chain, "IntensityUnits") ?? defaultUnits,
            Color: ReadColor(chain, "LightColor"),
            UseTemperature: ReadBool(chain, "bUseTemperature", false),
            Temperature: ReadDouble(chain, "Temperature", 6500.0),
            AttenuationRadius: ReadDouble(chain, "AttenuationRadius", 1000.0),
            SourceRadius: ReadDouble(chain, "SourceRadius", 0.0),
            SoftSourceRadius: ReadDouble(chain, "SoftSourceRadius", 0.0),
            SourceLength: ReadDouble(chain, "SourceLength", 0.0),
            InnerConeAngle: ReadDouble(chain, "InnerConeAngle", 0.0),
            OuterConeAngle: ReadDouble(chain, "OuterConeAngle", 44.0),
            UseInverseSquaredFalloff: ReadBool(chain, "bUseInverseSquaredFalloff", true),
            LightFalloffExponent: ReadDouble(chain, "LightFalloffExponent", 8.0),
            LightSourceAngle: ReadDouble(chain, "LightSourceAngle", 0.5357),
            IesTexture: ReadReference(chain, "IESTexture"),
            UseIesBrightness: ReadBool(chain, "bUseIESBrightness", false),
            IesBrightnessScale: ReadDouble(chain, "IESBrightnessScale", 1.0),
            LightFunctionMaterial: ReadReference(chain, "LightFunctionMaterial"),
            RealTimeCapture: ReadBool(chain, "bRealTimeCapture", false)
        );
    }

    private static LightComponentRecord[] ConvertKnownBlueprintLights(
        DefaultFileProvider provider,
        UObject actor
    )
    {
        if (actor.ExportType != "BP_HoodLight_C")
        {
            return [];
        }

        // Emitted as a fallback for instances that export no light components of
        // their own; the runtime drops these once concrete lights are present.
        const string packagePath = "Lighting/Library/Blueprints/BP_HoodLight.uasset";
        var objects = provider.LoadPackage(packagePath).GetExports().ToArray();
        var root = objects.FirstOrDefault(
            obj => obj.Name == "DefaultSceneRoot" + GeneratedVariableSuffix
        );
        var archetypes = new ComponentArchetypes(actor, root, objects, NoComponentTemplates);
        return objects
            .Where(obj => obj.ExportType.EndsWith("LightComponent", StringComparison.Ordinal))
            .OrderBy(obj => obj.Name, StringComparer.Ordinal)
            .Select(component => ConvertLightComponent(component, archetypes))
            .ToArray();
    }

    private static PostProcessRecord ConvertPostProcessVolume(UObject actor)
    {
        var settings = ReadStructFields(GetTaggedValue(actor, "Settings"));
        bool IsOverridden(string name) => ToBool(
            settings.GetValueOrDefault($"bOverride_{name}"),
            false
        );
        double? OverriddenDouble(string name) => IsOverridden(name)
            ? ToNullableDouble(settings.GetValueOrDefault(name))
            : null;
        string? OverriddenName(string name) => IsOverridden(name)
            ? JsonScalar(settings.GetValueOrDefault(name))?.ToString()
            : null;

        // Every field below is gated on its bOverride_ bit, matching UE's blend
        // rules: a value the artist typed and then unticked stays serialized in
        // the struct but never reaches the view. Conversely a ticked override
        // whose value equals the engine default is not serialized at all, so it
        // reads back as null -- Restir does exactly that for all five Film*
        // knobs, pinning the stock ACES curve.
        return new PostProcessRecord(
            Enabled: ReadBool(actor, "bEnabled", true),
            Unbound: ReadBool(actor, "bUnbound", false),
            Priority: ReadDouble(actor, "Priority", 0.0),
            BlendRadius: ReadDouble(actor, "BlendRadius", 100.0),
            BlendWeight: ReadDouble(actor, "BlendWeight", 1.0),
            BloomMethod: OverriddenName("BloomMethod"),
            BloomIntensity: OverriddenDouble("BloomIntensity"),
            FilmSlope: OverriddenDouble("FilmSlope"),
            FilmToe: OverriddenDouble("FilmToe"),
            FilmShoulder: OverriddenDouble("FilmShoulder"),
            FilmBlackClip: OverriddenDouble("FilmBlackClip"),
            FilmWhiteClip: OverriddenDouble("FilmWhiteClip"),
            AutoExposureMethod: IsOverridden("AutoExposureMethod")
                ? OverriddenName("AutoExposureMethod") ?? "AEM_Histogram"
                : null,
            AutoExposureMinEv100: OverriddenDouble("AutoExposureMinBrightness"),
            AutoExposureMaxEv100: OverriddenDouble("AutoExposureMaxBrightness"),
            AutoExposureBias: OverriddenDouble("AutoExposureBias")
        );
    }

    private static SkyAtmosphereComponentRecord ConvertSkyAtmosphereComponent(UObject component) =>
        new(
            Name: component.Name,
            Visible: ReadBool(component, "bVisible", true),
            HiddenInGame: ReadBool(component, "bHiddenInGame", false),
            TransformMode: ReadOptionalName(component, "TransformMode"),
            BottomRadiusKm: ReadOptionalDouble(component, "BottomRadius"),
            AtmosphereHeightKm: ReadOptionalDouble(component, "AtmosphereHeight"),
            GroundAlbedo: ReadOptionalColor(component, "GroundAlbedo"),
            RayleighScatteringScale: ReadOptionalDouble(component, "RayleighScatteringScale"),
            RayleighScatteringPerKm: ReadOptionalLinearColor(component, "RayleighScattering"),
            RayleighExponentialDistributionKm: ReadOptionalDouble(
                component,
                "RayleighExponentialDistribution"
            ),
            MieScatteringScale: ReadOptionalDouble(component, "MieScatteringScale"),
            MieScatteringPerKm: ReadOptionalLinearColor(component, "MieScattering"),
            MieAbsorptionScale: ReadOptionalDouble(component, "MieAbsorptionScale"),
            MieAbsorptionPerKm: ReadOptionalLinearColor(component, "MieAbsorption"),
            MieAnisotropy: ReadOptionalDouble(component, "MieAnisotropy"),
            MieExponentialDistributionKm: ReadOptionalDouble(
                component,
                "MieExponentialDistribution"
            ),
            OtherAbsorptionScale: ReadOptionalDouble(component, "OtherAbsorptionScale"),
            OtherAbsorptionPerKm: ReadOptionalLinearColor(component, "OtherAbsorption"),
            MultiScatteringFactor: ReadOptionalDouble(component, "MultiScatteringFactor"),
            SkyLuminanceFactor: ReadOptionalLinearColor(component, "SkyLuminanceFactor"),
            SkyAndAerialPerspectiveLuminanceFactor: ReadOptionalLinearColor(
                component,
                "SkyAndAerialPerspectiveLuminanceFactor"
            ),
            AerialPerspectiveStartDepthKm: ReadOptionalDouble(
                component,
                "AerialPerspectiveStartDepth"
            ),
            AerialPerspectiveViewDistanceScale: ReadOptionalDouble(
                component,
                // The misspelling is part of Unreal's serialized property name.
                "AerialPespectiveViewDistanceScale"
            ),
            HeightFogContribution: ReadOptionalDouble(component, "HeightFogContribution")
        );

    private static HeightFogComponentRecord ConvertHeightFogComponent(UObject component) =>
        new(
            Name: component.Name,
            Visible: ReadBool(component, "bVisible", true),
            HiddenInGame: ReadBool(component, "bHiddenInGame", false),
            FogDensity: ReadOptionalDouble(component, "FogDensity"),
            FogHeightFalloff: ReadOptionalDouble(component, "FogHeightFalloff"),
            FogInscatteringColor: ReadOptionalLinearColor(component, "FogInscatteringLuminance"),
            FogMaxOpacity: ReadOptionalDouble(component, "FogMaxOpacity"),
            StartDistanceCm: ReadOptionalDouble(component, "StartDistance"),
            EndDistanceCm: ReadOptionalDouble(component, "EndDistance"),
            FogCutoffDistanceCm: ReadOptionalDouble(component, "FogCutoffDistance"),
            DirectionalInscatteringColor: ReadOptionalLinearColor(
                component,
                "DirectionalInscatteringLuminance"
            ),
            DirectionalInscatteringExponent: ReadOptionalDouble(
                component,
                "DirectionalInscatteringExponent"
            ),
            DirectionalInscatteringStartDistanceCm: ReadOptionalDouble(
                component,
                "DirectionalInscatteringStartDistance"
            ),
            EnableVolumetricFog: ReadOptionalBool(component, "bEnableVolumetricFog"),
            VolumetricFogAlbedo: ReadOptionalColor(component, "VolumetricFogAlbedo"),
            VolumetricFogEmissive: ReadOptionalLinearColor(component, "VolumetricFogEmissive"),
            VolumetricFogExtinctionScale: ReadOptionalDouble(
                component,
                "VolumetricFogExtinctionScale"
            ),
            VolumetricFogScatteringDistribution: ReadOptionalDouble(
                component,
                "VolumetricFogScatteringDistribution"
            ),
            VolumetricFogStartDistanceCm: ReadOptionalDouble(
                component,
                "VolumetricFogStartDistance"
            ),
            VolumetricFogNearFadeInDistanceCm: ReadOptionalDouble(
                component,
                "VolumetricFogNearFadeInDistance"
            ),
            VolumetricFogDistanceCm: ReadOptionalDouble(component, "VolumetricFogDistance")
        );

    private static object? ReadComponentValue(UObject component, string name) =>
        GetTaggedValue(component, name) ?? GetPublicMember(component, name);

    // Archetype reads never consult CUE4Parse's typed members: those hold the
    // parser's own placeholders (pi intensity for every light class) rather than
    // the UE class defaults an absent delta implies.
    private static object? ReadTaggedValue(UObject[] chain, string name)
    {
        foreach (var component in chain)
        {
            var value = GetTaggedValue(component, name);
            if (value is not null)
            {
                return value;
            }
        }
        return null;
    }

    private static T ReadStruct<T>(UObject[] chain, string name, T defaultValue)
    {
        foreach (var component in chain)
        {
            if (GetTaggedValue(component, name) is not null)
            {
                return component.GetOrDefault(name, defaultValue);
            }
        }
        return defaultValue;
    }

    private static FTransform ReadRelativeTransform(UObject[] chain) => new(
        ReadStruct(chain, "RelativeRotation", FRotator.ZeroRotator),
        ReadStruct(chain, "RelativeLocation", FVector.ZeroVector),
        ReadStruct(chain, "RelativeScale3D", FVector.OneVector)
    );

    private static double ReadDouble(UObject[] chain, string name, double defaultValue) =>
        ToNullableDouble(ReadTaggedValue(chain, name)) ?? defaultValue;

    private static bool ReadBool(UObject[] chain, string name, bool defaultValue) =>
        ToBool(ReadTaggedValue(chain, name), defaultValue);

    private static string? ReadName(UObject[] chain, string name)
    {
        var value = ReadTaggedValue(chain, name);
        return value is null
            ? null
            : GetPublicMember(value, "Text")?.ToString() ?? value.ToString();
    }

    private static string? ReadReference(UObject[] chain, string name) =>
        ReadTaggedValue(chain, name) is FPackageIndex index
            ? index.ResolvedObject?.GetPathName()
            : null;

    private static ColorRecord ReadColor(UObject[] chain, string name)
    {
        var value = ReadTaggedValue(chain, name);
        value = GetPublicMember(value, "StructType") ?? value;
        return new ColorRecord(
            R: ReadByteMember(value, "R", 255),
            G: ReadByteMember(value, "G", 255),
            B: ReadByteMember(value, "B", 255),
            A: ReadByteMember(value, "A", 255)
        );
    }

    private static double ReadDouble(UObject component, string name, double defaultValue)
    {
        var value = ReadComponentValue(component, name);
        if (value is null)
        {
            return defaultValue;
        }
        try
        {
            return Convert.ToDouble(value, CultureInfo.InvariantCulture);
        }
        catch (Exception) when (value is not string)
        {
            return defaultValue;
        }
    }

    private static double? ReadOptionalDouble(UObject component, string name) =>
        ToNullableDouble(ReadComponentValue(component, name));

    private static bool? ReadOptionalBool(UObject component, string name)
    {
        var value = ReadComponentValue(component, name);
        return value is null ? null : ToBool(value, false);
    }

    private static string? ReadOptionalName(UObject component, string name) =>
        ReadComponentValue(component, name) is null ? null : ReadName(component, name);

    private static LinearColorRecord? ReadOptionalLinearColor(UObject component, string name)
    {
        var value = ReadComponentValue(component, name);
        if (value is null)
        {
            return null;
        }
        value = GetPublicMember(value, "StructType") ?? value;
        return new LinearColorRecord(
            R: ReadDoubleMember(value, "R", 1.0),
            G: ReadDoubleMember(value, "G", 1.0),
            B: ReadDoubleMember(value, "B", 1.0),
            A: ReadDoubleMember(value, "A", 1.0)
        );
    }

    private static ColorRecord? ReadOptionalColor(UObject component, string name) =>
        ReadComponentValue(component, name) is null ? null : ReadColor(component, name);

    private static double ReadDoubleMember(object? value, string name, double defaultValue)
    {
        var member = GetPublicMember(value, name);
        return ToNullableDouble(member) ?? defaultValue;
    }

    private static bool ReadBool(UObject component, string name, bool defaultValue) =>
        ToBool(ReadComponentValue(component, name), defaultValue);

    private static bool ToBool(object? value, bool defaultValue) => value switch
        {
            bool typed => typed,
            byte typed => typed != 0,
            sbyte typed => typed != 0,
            short typed => typed != 0,
            ushort typed => typed != 0,
            int typed => typed != 0,
            uint typed => typed != 0,
            long typed => typed != 0,
            ulong typed => typed != 0,
            _ => defaultValue,
        };

    private static double? ToNullableDouble(object? value)
    {
        if (value is null)
        {
            return null;
        }
        try
        {
            return Convert.ToDouble(value, CultureInfo.InvariantCulture);
        }
        catch (Exception error) when (
            error is FormatException or InvalidCastException or OverflowException
        )
        {
            return null;
        }
    }

    private static string? ReadName(UObject component, string name)
    {
        var value = ReadComponentValue(component, name);
        return GetPublicMember(value, "Text")?.ToString() ?? value?.ToString();
    }

    private static ColorRecord ReadColor(UObject component, string name)
    {
        var value = ReadComponentValue(component, name);
        value = GetPublicMember(value, "StructType") ?? value;
        return new ColorRecord(
            R: ReadByteMember(value, "R", 255),
            G: ReadByteMember(value, "G", 255),
            B: ReadByteMember(value, "B", 255),
            A: ReadByteMember(value, "A", 255)
        );
    }

    private static byte ReadByteMember(object? value, string name, byte defaultValue)
    {
        var member = GetPublicMember(value, name);
        if (member is null)
        {
            return defaultValue;
        }
        try
        {
            return Convert.ToByte(member, CultureInfo.InvariantCulture);
        }
        catch
        {
            return defaultValue;
        }
    }

    // Blueprint components are delta-serialized against their Simple Construction
    // Script template, so every property has to resolve instance tag -> template
    // tag -> UE class default.
    private static Dictionary<string, UObject> LoadBlueprintComponentTemplates(
        DefaultFileProvider provider,
        string? actorClassPath
    )
    {
        if (actorClassPath is null ||
            !actorClassPath.StartsWith("/Game/", StringComparison.Ordinal))
        {
            return NoComponentTemplates;
        }
        if (BlueprintComponentTemplates.TryGetValue(actorClassPath, out var templates))
        {
            return templates;
        }
        templates = new Dictionary<string, UObject>(StringComparer.Ordinal);
        var packageKey = ObjectPathToPackageKey(actorClassPath);
        if (provider.Files.ContainsKey(packageKey))
        {
            foreach (var template in provider.LoadPackage(packageKey).GetExports()
                .Where(export => export.Name.EndsWith(
                    GeneratedVariableSuffix,
                    StringComparison.Ordinal
                )))
            {
                var variableName = template.Name[..^GeneratedVariableSuffix.Length];
                if (!templates.TryAdd(variableName, template))
                {
                    throw new InvalidDataException(
                        $"blueprint {actorClassPath} has duplicate " +
                        $"component template {variableName}"
                    );
                }
            }
        }
        BlueprintComponentTemplates.Add(actorClassPath, templates);
        return templates;
    }

    private sealed class ComponentArchetypes
    {
        private readonly Dictionary<string, UObject> components = new(StringComparer.Ordinal);
        private readonly Dictionary<string, UObject> templates;
        private readonly UObject? root;

        public ComponentArchetypes(
            UObject? actor,
            UObject? rootComponent,
            IEnumerable<UObject> ownedObjects,
            Dictionary<string, UObject> templates
        )
        {
            foreach (var component in ownedObjects)
            {
                components.TryAdd(component.Name, component);
            }
            this.templates = templates;
            root = rootComponent;
            ActorType = actor?.ExportType;
            ActorClass = actor?.Class?.GetPathName();
        }

        public string? ActorType { get; }

        public string? ActorClass { get; }

        public UObject[] Chain(UObject component)
        {
            var template = templates.GetValueOrDefault(component.Name);
            return template is null || ReferenceEquals(template, component)
                ? [component]
                : [component, template];
        }

        public bool IsRoot(UObject component) =>
            ReferenceEquals(component, root) || component.Name == root?.Name;

        // Components serialize their transform relative to AttachParent; the
        // manifest stores it relative to the actor root instead.
        public TransformRecord TransformRelativeToActor(UObject component)
        {
            if (IsRoot(component))
            {
                return ConvertTransform(FTransform.Identity);
            }
            var transform = ConvertTransform(ReadRelativeTransform(Chain(component)));
            var visited = new HashSet<string>(StringComparer.Ordinal) { component.Name };
            for (
                var parent = AttachParent(component);
                parent is not null && !IsRoot(parent);
                parent = AttachParent(parent)
            )
            {
                if (!visited.Add(parent.Name))
                {
                    throw new InvalidDataException(
                        $"component {component.Name} has a cyclic AttachParent chain"
                    );
                }
                transform = ComposeTransforms(
                    ConvertTransform(ReadRelativeTransform(Chain(parent))),
                    transform
                );
            }
            return transform;
        }

        private UObject? AttachParent(UObject component)
        {
            var name = (ReadTaggedValue(Chain(component), "AttachParent") as FPackageIndex)
                ?.ResolvedObject?.Name.Text;
            if (name is null)
            {
                return null;
            }
            return components.GetValueOrDefault(name)
                ?? (name.EndsWith(GeneratedVariableSuffix, StringComparison.Ordinal)
                    ? components.GetValueOrDefault(name[..^GeneratedVariableSuffix.Length])
                    : null);
        }
    }

    private static TransformRecord ComposeTransforms(
        TransformRecord parent,
        TransformRecord child
    )
    {
        var offset = RotateVector(parent.Rotation, new Vec3Record(
            parent.Scale.X * child.Translation.X,
            parent.Scale.Y * child.Translation.Y,
            parent.Scale.Z * child.Translation.Z
        ));
        return new TransformRecord(
            Translation: new Vec3Record(
                parent.Translation.X + offset.X,
                parent.Translation.Y + offset.Y,
                parent.Translation.Z + offset.Z
            ),
            Rotation: MultiplyQuaternions(parent.Rotation, child.Rotation),
            Scale: new Vec3Record(
                parent.Scale.X * child.Scale.X,
                parent.Scale.Y * child.Scale.Y,
                parent.Scale.Z * child.Scale.Z
            )
        );
    }

    private static QuatRecord MultiplyQuaternions(QuatRecord parent, QuatRecord child) => new(
        X: parent.W * child.X + parent.X * child.W + parent.Y * child.Z - parent.Z * child.Y,
        Y: parent.W * child.Y - parent.X * child.Z + parent.Y * child.W + parent.Z * child.X,
        Z: parent.W * child.Z + parent.X * child.Y - parent.Y * child.X + parent.Z * child.W,
        W: parent.W * child.W - parent.X * child.X - parent.Y * child.Y - parent.Z * child.Z
    );

    private static Vec3Record RotateVector(QuatRecord rotation, Vec3Record value)
    {
        var crossX = 2.0 * (rotation.Y * value.Z - rotation.Z * value.Y);
        var crossY = 2.0 * (rotation.Z * value.X - rotation.X * value.Z);
        var crossZ = 2.0 * (rotation.X * value.Y - rotation.Y * value.X);
        return new Vec3Record(
            value.X + rotation.W * crossX + rotation.Y * crossZ - rotation.Z * crossY,
            value.Y + rotation.W * crossY + rotation.Z * crossX - rotation.X * crossZ,
            value.Z + rotation.W * crossZ + rotation.X * crossY - rotation.Y * crossX
        );
    }

    private static string? ActorPackageDirectory(string? classPath)
    {
        if (classPath is null || !classPath.StartsWith("/Game/", StringComparison.Ordinal))
        {
            return null;
        }
        var packagePath = classPath["/Game/".Length..];
        var slash = packagePath.LastIndexOf('/');
        return slash < 0 ? null : packagePath[..(slash + 1)];
    }

    private static TransformRecord ConvertTransform(FTransform transform) => new(
        Translation: new Vec3Record(
            transform.Translation.X,
            transform.Translation.Y,
            transform.Translation.Z
        ),
        Rotation: new QuatRecord(
            transform.Rotation.X,
            transform.Rotation.Y,
            transform.Rotation.Z,
            transform.Rotation.W
        ),
        Scale: new Vec3Record(
            transform.Scale3D.X,
            transform.Scale3D.Y,
            transform.Scale3D.Z
        )
    );

    private static void Increment(Dictionary<string, int> counts, string key)
    {
        counts.TryGetValue(key, out var count);
        counts[key] = count + 1;
    }

    private static string Format(Vec3Record value) =>
        $"({value.X:R},{value.Y:R},{value.Z:R})";

    private static string Format(QuatRecord value) =>
        $"({value.X:R},{value.Y:R},{value.Z:R},{value.W:R})";

    private static void DumpShape(object value)
    {
        foreach (var property in value.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public))
        {
            if (property.GetIndexParameters().Length != 0)
            {
                continue;
            }

            object? propertyValue;
            try
            {
                propertyValue = property.GetValue(value);
            }
            catch (Exception error)
            {
                Console.WriteLine(
                    $"ZORAH_SHAPE property={property.Name} type={property.PropertyType.FullName} " +
                    $"error={error.GetType().Name}"
                );
                continue;
            }

            Console.WriteLine(
                $"ZORAH_SHAPE property={property.Name} type={property.PropertyType.FullName} " +
                $"value={Describe(propertyValue)}"
            );
        }
    }

    private static void DumpMembers(object value)
    {
        foreach (var field in value.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public))
        {
            object? fieldValue;
            try
            {
                fieldValue = field.GetValue(value);
            }
            catch (Exception error)
            {
                fieldValue = $"error:{error.GetType().Name}";
            }
            Console.WriteLine(
                $"ZORAH_MEMBER field={field.Name} type={field.FieldType.FullName} " +
                $"value={Describe(fieldValue)}"
            );
        }
        DumpShape(value);
    }

    private static string Describe(object? value) => value switch
    {
        null => "null",
        string text => text,
        Array array => $"array[{array.Length}]",
        ICollection collection => $"collection[{collection.Count}]",
        ValueType => value.ToString() ?? value.GetType().Name,
        _ => value.GetType().FullName ?? value.GetType().Name,
    };

    private static void DumpNested(
        string path,
        object? value,
        int depth,
        HashSet<object> visited
    )
    {
        Console.WriteLine(
            $"ZORAH_NESTED path={path} type={value?.GetType().FullName ?? "null"} " +
            $"value={Describe(value)}"
        );
        if (value is null || depth >= 6 || value is string || value.GetType().IsPrimitive ||
            value.GetType().IsEnum || value is decimal)
        {
            return;
        }
        if (value is FPackageIndex packageIndex)
        {
            Console.WriteLine(
                $"ZORAH_NESTED_REFERENCE path={path} " +
                $"object={packageIndex.ResolvedObject?.GetPathName() ?? "None"}"
            );
            return;
        }
        if (!value.GetType().IsValueType && !visited.Add(value))
        {
            return;
        }
        if (value is IEnumerable enumerable)
        {
            var index = 0;
            foreach (var item in enumerable)
            {
                if (index == 100)
                {
                    Console.WriteLine($"ZORAH_NESTED_TRUNCATED path={path}");
                    break;
                }
                DumpNested($"{path}[{index}]", item, depth + 1, visited);
                index++;
            }
            return;
        }

        foreach (var field in value.GetType().GetFields(BindingFlags.Instance | BindingFlags.Public))
        {
            DumpNested($"{path}.{field.Name}", field.GetValue(value), depth + 1, visited);
        }
        foreach (var property in value.GetType().GetProperties(BindingFlags.Instance | BindingFlags.Public))
        {
            if (property.GetIndexParameters().Length != 0 || property.Name is "GenericValue")
            {
                continue;
            }
            object? nested;
            try
            {
                nested = property.GetValue(value);
            }
            catch
            {
                continue;
            }
            DumpNested($"{path}.{property.Name}", nested, depth + 1, visited);
        }
    }
}

sealed record SceneManifest(
    string Format,
    string EngineVersion,
    string Level,
    string SourceMap,
    string ExternalActorPrefix,
    int ActorPackageCount,
    int PackagesWithoutActors,
    Dictionary<string, int> ActorTypeCounts,
    Dictionary<string, int> StaticMeshComponentTypeCounts,
    int UnresolvedStaticMeshComponents,
    string[] ReferencedMeshes,
    ActorRecord[] Actors,
    FailureRecord[] Failures
);

sealed record MeshMaterialManifest(
    string Format,
    string EngineVersion,
    MeshMaterialRecord[] Meshes,
    FailureRecord[] Failures
);

sealed record MeshMaterialRecord(
    string Object,
    string Package,
    MeshMaterialSlotRecord[] Slots,
    MeshSectionMaterialRecord[] Sections
);

sealed record MeshMaterialSlotRecord(
    int Index,
    string? Material,
    string? SlotName,
    string? ImportedSlotName
);

sealed record MeshSectionMaterialRecord(
    int Lod,
    int Section,
    int MaterialIndex
);

sealed record ActorRecord(
    string Package,
    string Name,
    string? Label,
    string Type,
    string? Class,
    TransformRecord Transform,
    bool Hidden,
    StaticMeshComponentRecord[] Components,
    LightComponentRecord[] Lights,
    SkyAtmosphereComponentRecord? Atmosphere,
    HeightFogComponentRecord? HeightFog,
    PostProcessRecord? PostProcess
);

sealed record SkyAtmosphereComponentRecord(
    string Name,
    bool Visible,
    bool HiddenInGame,
    string? TransformMode,
    double? BottomRadiusKm,
    double? AtmosphereHeightKm,
    ColorRecord? GroundAlbedo,
    double? RayleighScatteringScale,
    LinearColorRecord? RayleighScatteringPerKm,
    double? RayleighExponentialDistributionKm,
    double? MieScatteringScale,
    LinearColorRecord? MieScatteringPerKm,
    double? MieAbsorptionScale,
    LinearColorRecord? MieAbsorptionPerKm,
    double? MieAnisotropy,
    double? MieExponentialDistributionKm,
    double? OtherAbsorptionScale,
    LinearColorRecord? OtherAbsorptionPerKm,
    double? MultiScatteringFactor,
    LinearColorRecord? SkyLuminanceFactor,
    LinearColorRecord? SkyAndAerialPerspectiveLuminanceFactor,
    double? AerialPerspectiveStartDepthKm,
    double? AerialPerspectiveViewDistanceScale,
    double? HeightFogContribution
);

sealed record HeightFogComponentRecord(
    string Name,
    bool Visible,
    bool HiddenInGame,
    double? FogDensity,
    double? FogHeightFalloff,
    LinearColorRecord? FogInscatteringColor,
    double? FogMaxOpacity,
    double? StartDistanceCm,
    double? EndDistanceCm,
    double? FogCutoffDistanceCm,
    LinearColorRecord? DirectionalInscatteringColor,
    double? DirectionalInscatteringExponent,
    double? DirectionalInscatteringStartDistanceCm,
    bool? EnableVolumetricFog,
    ColorRecord? VolumetricFogAlbedo,
    LinearColorRecord? VolumetricFogEmissive,
    double? VolumetricFogExtinctionScale,
    double? VolumetricFogScatteringDistribution,
    double? VolumetricFogStartDistanceCm,
    double? VolumetricFogNearFadeInDistanceCm,
    double? VolumetricFogDistanceCm
);

sealed record PostProcessRecord(
    bool Enabled,
    bool Unbound,
    double Priority,
    double BlendRadius,
    double BlendWeight,
    string? BloomMethod,
    double? BloomIntensity,
    double? FilmSlope,
    double? FilmToe,
    double? FilmShoulder,
    double? FilmBlackClip,
    double? FilmWhiteClip,
    string? AutoExposureMethod,
    double? AutoExposureMinEv100,
    double? AutoExposureMaxEv100,
    double? AutoExposureBias
);

sealed record LightComponentRecord(
    string Name,
    string Type,
    string ComponentType,
    TransformRecord Transform,
    bool Visible,
    bool HiddenInGame,
    bool AffectsWorld,
    bool CastShadows,
    double Intensity,
    string IntensityUnits,
    ColorRecord Color,
    bool UseTemperature,
    double Temperature,
    double AttenuationRadius,
    double SourceRadius,
    double SoftSourceRadius,
    double SourceLength,
    double InnerConeAngle,
    double OuterConeAngle,
    bool UseInverseSquaredFalloff,
    double LightFalloffExponent,
    double LightSourceAngle,
    string? IesTexture,
    bool UseIesBrightness,
    double IesBrightnessScale,
    string? LightFunctionMaterial,
    bool RealTimeCapture
);

sealed record StaticMeshComponentRecord(
    string Name,
    string Type,
    string? Template,
    string? Mesh,
    string? MeshResolution,
    TransformRecord Transform,
    bool Visible,
    bool HiddenInGame,
    bool CastShadow,
    string[] OverrideMaterials,
    TransformRecord[]? Instances,
    string? MissingReason
);

sealed record TransformRecord(
    Vec3Record Translation,
    QuatRecord Rotation,
    Vec3Record Scale
);

sealed record Vec3Record(double X, double Y, double Z);
sealed record QuatRecord(double X, double Y, double Z, double W);
sealed record ColorRecord(byte R, byte G, byte B, byte A);
sealed record LinearColorRecord(double R, double G, double B, double A);
sealed record FailureRecord(string Package, string ErrorType, string Message);

sealed record MaterialManifest(
    string Format,
    string EngineVersion,
    string[] Requested,
    MaterialRecord[] Materials,
    string[] TextureReferences,
    FailureRecord[] Failures
);

sealed record MaterialRecord(
    string Package,
    string Object,
    string Type,
    string? Parent,
    MaterialParameterRecord[] Scalars,
    MaterialParameterRecord[] Vectors,
    MaterialParameterRecord[] Textures,
    StaticSwitchParameterRecord[] StaticSwitches,
    string[] Layers,
    string[] Blends,
    Dictionary<string, object?> BaseOverrides
);

sealed record MaterialParameterRecord(
    string Name,
    string? Association,
    int? Index,
    object? Value
);

sealed record StaticSwitchParameterRecord(
    string Name,
    string? Association,
    int? Index,
    bool Value,
    bool Override
);

sealed record MaterialLayerFunctions(string[] Layers, string[] Blends);

sealed record MaterialExpressionDefaults(
    MaterialParameterRecord[] Scalars,
    MaterialParameterRecord[] Vectors,
    MaterialParameterRecord[] Textures,
    StaticSwitchParameterRecord[] StaticSwitches
);

sealed record TextureManifest(
    string Format,
    string EngineVersion,
    TextureRecord[] Textures,
    FailureRecord[] Failures
);

sealed record TextureRecord(
    string Object,
    string Package,
    string Output,
    int Width,
    int Height,
    string PixelFormat,
    string? SourceCompression,
    bool Srgb,
    bool IsNormalMap,
    bool EditorSource,
    long PayloadSize,
    bool Exported,
    TextureBlockRecord[] Blocks
);

sealed record TextureBlockRecord(
    int BlockX,
    int BlockY,
    int Width,
    int Height,
    long PayloadOffset,
    long PayloadSize
);
