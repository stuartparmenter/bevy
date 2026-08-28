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
    // Slots the runtime binds by parameter name, keyed by the material input the
    // graph walk follows to reach them. Emitting the runtime name rather than
    // the authoring one lets a connection-derived texture join the same
    // selection the parameter families already drive.
    private static readonly (string Input, string Parameter)[] MaterialGraphTextureInputs =
    [
        ("BaseColor", "Base Color"),
        ("Normal", "Normal"),
        ("EmissiveColor", "Emissive"),
    ];
    private const string MaterialGraphOrmParameter = "ORM";
    // UE's roughness input and the runtime scalar that scales the packed map
    // share this one name.
    private const string RoughnessInput = "Roughness";
    // UE packs occlusion, roughness and metalness into one texture's R, G and B,
    // which a MaterialExpressionTextureSample exposes as output indices 1, 2 and
    // 3. Requiring every connected input to read its own channel of one shared
    // sample is what separates a packed ORM map from three unrelated textures;
    // Specular is deliberately absent because a StandardMaterial has no map for
    // it, only the scalar reflectance.
    private static readonly (string Input, int OutputIndex)[] MaterialGraphOrmInputs =
    [
        ("AmbientOcclusion", 1),
        (RoughnessInput, 2),
        ("Metallic", 3),
    ];
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
            var blockSize = 1 << blockSizeExponent;
            ulong written = 0;
            // Scoped so the handle is closed before the move; Windows refuses
            // to rename a file that is still open.
            await using (var output = File.Create(temporaryPath))
            {
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
                    throw new InvalidDataException(
                        $"raw-source wrote {written} bytes; expected {rawSize}"
                    );
                }
                await output.FlushAsync();
            }
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

    /// The asset name of an object path, without package path or object suffix.
    private static string ShortObjectName(string path)
    {
        var name = path;
        var dot = name.LastIndexOf('.');
        if (dot >= 0)
        {
            name = name[(dot + 1)..];
        }
        var slash = name.LastIndexOf('/');
        return slash >= 0 ? name[(slash + 1)..] : name;
    }

    private static string? NameText(object? value) => GetPublicMember(value, "Text") as string;

    /// An enum tag's value without its type prefix: EDataLayerType::Runtime is Runtime.
    private static string? EnumValueName(object? value)
    {
        var text = NameText(value) ?? value as string ?? value?.ToString();
        if (string.IsNullOrEmpty(text) || text == "None")
        {
            return null;
        }
        var separator = text.LastIndexOf("::", StringComparison.Ordinal);
        return separator >= 0 ? text[(separator + 2)..] : text;
    }

    // A TSoftObjectPtr serializes as an FSoftObjectPath; a hard reference to the
    // same asset arrives as an FPackageIndex instead.
    private static string? ObjectReferencePath(object? value)
    {
        if (PackageReferencePath(value) is string reference)
        {
            return reference;
        }
        var soft = StructValue(value);
        if (soft is null)
        {
            return null;
        }
        var assetPath = GetPublicMember(soft, "AssetPath");
        var packageName = NameText(GetPublicMember(assetPath, "PackageName"));
        var assetName = NameText(GetPublicMember(assetPath, "AssetName"));
        if (!string.IsNullOrEmpty(packageName) && !string.IsNullOrEmpty(assetName))
        {
            return $"{packageName}.{assetName}";
        }
        var path = NameText(GetPublicMember(soft, "AssetPathName"))
            ?? (soft.GetType().Name.Contains("SoftObjectPath", StringComparison.Ordinal)
                ? soft.ToString()
                : soft as string);
        return string.IsNullOrEmpty(path) || path == "None" ? null : path;
    }

    /// The data layer assets an actor belongs to, as object paths.
    ///
    /// World Partition actors reference layer assets through DataLayerAssets;
    /// pre-5.0 actors carry bare FActorDataLayer names in DataLayers instead, so
    /// an entry is not always a path.
    private static string[] ReadActorDataLayers(UObject actor)
    {
        var layers = new List<string>();
        foreach (var entry in ReadArrayValues(GetTaggedValue(actor, "DataLayerAssets")))
        {
            if (ObjectReferencePath(entry) is string path)
            {
                layers.Add(path);
            }
        }
        foreach (var entry in ReadArrayValues(GetTaggedValue(actor, "DataLayers")))
        {
            var name = NameText(ReadStructFields(entry).GetValueOrDefault("Name"));
            if (!string.IsNullOrEmpty(name) && name != "None")
            {
                layers.Add(name);
            }
        }
        return layers.Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
    }

    private static UObject? ResolveExport(object? value, UObject[] packageObjects)
    {
        if (value is not FPackageIndex index)
        {
            return null;
        }
        var name = index.ResolvedObject?.Name.Text;
        if (name is not null &&
            packageObjects.FirstOrDefault(export => export.Name == name) is UObject named)
        {
            return named;
        }
        return index.Index > 0 && index.Index <= packageObjects.Length
            ? packageObjects[index.Index - 1]
            : null;
    }

    /// The layer definitions a level's WorldDataLayers actor declares, by short name.
    ///
    /// A UDataLayerInstance leaves InitialRuntimeState and the editor flags
    /// untagged while they hold their class default, so an absent tag means the
    /// UE default - Unloaded, visible, loaded - not an unknown value.
    private static Dictionary<string, DataLayerRecord> ReadWorldDataLayers(
        DefaultFileProvider provider,
        UObject worldDataLayers,
        UObject[] packageObjects
    )
    {
        var records = new Dictionary<string, DataLayerRecord>(StringComparer.Ordinal);
        foreach (var entry in ReadArrayValues(
            GetTaggedValue(worldDataLayers, "DataLayerInstances")
        ))
        {
            if (ResolveExport(entry, packageObjects) is not UObject instance)
            {
                continue;
            }
            var asset = ObjectReferencePath(GetTaggedValue(instance, "DataLayerAsset"));
            var name = asset is null ? instance.Name : ShortObjectName(asset);
            records[name] = new DataLayerRecord(
                Name: name,
                Asset: asset,
                Type: ReadDataLayerType(provider, asset),
                InitialRuntimeState: EnumValueName(GetTaggedValue(instance, "InitialRuntimeState"))
                    ?? "Unloaded",
                InitiallyVisible: instance.GetOrDefault("bIsInitiallyVisible", true),
                InitiallyLoadedInEditor: instance.GetOrDefault("bIsInitiallyLoadedInEditor", true)
            );
        }
        return records;
    }

    private static string ReadDataLayerType(DefaultFileProvider provider, string? assetPath)
    {
        if (assetPath is null || !assetPath.Contains('/'))
        {
            return "Unknown";
        }
        var key = ObjectPathToPackageKey(assetPath);
        if (!provider.Files.ContainsKey(key))
        {
            return "Unknown";
        }
        try
        {
            var asset = provider.LoadPackage(key).GetExports()
                .FirstOrDefault(export => export.ExportType == "DataLayerAsset");
            return (asset is null ? null : EnumValueName(GetTaggedValue(asset, "DataLayerType")))
                ?? "Unknown";
        }
        catch
        {
            return "Unknown";
        }
    }

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
                Value: JsonScalar(fields.GetValueOrDefault("ParameterValue")),
                ExpressionGuid: ReadGuid(fields.GetValueOrDefault("ExpressionGUID"))
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
                Override: ToBool(fields.GetValueOrDefault("bOverride"), false),
                ExpressionGuid: ReadGuid(fields.GetValueOrDefault("ExpressionGUID"))
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
        var owned = exports
            .Where(candidate =>
                candidate.GetPathName().StartsWith(materialPrefix, StringComparison.Ordinal))
            .ToArray();
        var scalars = new List<MaterialParameterRecord>();
        var vectors = new List<MaterialParameterRecord>();
        var textures = new List<MaterialParameterRecord>();
        var staticSwitches = new List<StaticSwitchParameterRecord>();
        foreach (var expression in owned.Where(candidate =>
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
                Value: null,
                ExpressionGuid: ReadGuid(GetTaggedValue(expression, "ExpressionGUID"))
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
                    Override: false,
                    ExpressionGuid: parameter.ExpressionGuid
                ));
            }
        }
        var graph = ReadMaterialGraphDefaults(material, owned);
        return new MaterialExpressionDefaults(
            MergeMaterialParameters(graph.Scalars, scalars),
            MergeMaterialParameters([], vectors),
            // Named parameters win over the graph walk: an instance can override
            // them by name, while a slot resolved through the graph is fixed in
            // the base material and can only ever carry its authored texture.
            MergeMaterialParameters(graph.Textures, textures),
            MergeStaticSwitchParameters([], staticSwitches)
        );
    }

    // A base UMaterial keeps its graph in the MaterialEditorOnlyData export:
    // every shading input is an FExpressionInput naming the expression it reads
    // and the output index it takes off that expression. Nodes reached that way
    // are frequently plain MaterialExpressionTextureSamples with no
    // ParameterName, which the loop above skips, so a material that samples its
    // maps directly rather than through parameters otherwise contributes no
    // textures at all. Only unnamed samples are emitted here; a named one is
    // already recorded under the name instances override it by.
    private static MaterialGraphDefaults ReadMaterialGraphDefaults(
        UObject material,
        UObject[] owned
    )
    {
        var editorOnlyData = owned.FirstOrDefault(candidate =>
            candidate.ExportType.Equals("MaterialEditorOnlyData", StringComparison.Ordinal));
        if (editorOnlyData is null)
        {
            return new MaterialGraphDefaults([], []);
        }
        var expressions = owned
            .Where(candidate =>
                candidate.ExportType.StartsWith("MaterialExpression", StringComparison.Ordinal))
            .ToDictionary(candidate => candidate.Name, StringComparer.Ordinal);
        // CustomizedUVs serializes once per channel; every input read below is
        // written at most once, so the first tag is always the whole value.
        var inputs = editorOnlyData.Properties
            .GroupBy(property => property.Name.Text, StringComparer.Ordinal)
            .ToDictionary(
                group => group.Key,
                group => group.First().Tag?.GenericValue,
                StringComparer.Ordinal
            );
        var resolved = inputs.ToDictionary(
            pair => pair.Key,
            pair => ResolveGraphTexture(pair.Value, expressions),
            StringComparer.Ordinal
        );

        var mapped = new HashSet<string>(StringComparer.Ordinal);
        var scalars = new List<MaterialParameterRecord>();
        var textures = new List<MaterialParameterRecord>();
        foreach (var (input, parameter) in MaterialGraphTextureInputs)
        {
            if (resolved.GetValueOrDefault(input) is not
                { ParameterName: null, Reference: not null } texture)
            {
                continue;
            }
            mapped.Add(input);
            textures.Add(new MaterialParameterRecord(parameter, "GlobalParameter", -1, texture.Reference));
        }

        var packed = MaterialGraphOrmInputs
            .Select(entry => (entry.Input, entry.OutputIndex, Texture: resolved.GetValueOrDefault(entry.Input)))
            .Where(entry => entry.Texture is { ParameterName: null, Reference: not null })
            .ToArray();
        if (packed.Length != 0 && packed.All(entry =>
            entry.Texture!.OutputIndex == entry.OutputIndex &&
            string.Equals(entry.Texture.Reference, packed[0].Texture!.Reference, StringComparison.Ordinal)))
        {
            mapped.UnionWith(packed.Select(entry => entry.Input));
            textures.Add(new MaterialParameterRecord(
                MaterialGraphOrmParameter,
                "GlobalParameter",
                -1,
                packed[0].Texture!.Reference
            ));
            if (ReadGraphRoughnessScale(inputs.GetValueOrDefault(RoughnessInput), expressions)
                is { } scale)
            {
                scalars.Add(new MaterialParameterRecord(RoughnessInput, "GlobalParameter", -1, scale));
            }
        }

        var objectPath = material.GetPathName();
        foreach (var parameter in scalars.Concat(textures))
        {
            Console.WriteLine(
                $"ZORAH_MATERIAL_GRAPH_DEFAULT object={objectPath} " +
                $"value={parameter.Value} parameter={parameter.Name}"
            );
        }
        foreach (var (input, texture) in resolved.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            if (mapped.Contains(input) || texture is not { ParameterName: null, Reference: not null })
            {
                continue;
            }
            var reason = MaterialGraphOrmInputs.Any(entry => entry.Input == input)
                ? "not-a-packed-orm-sample"
                : "no-runtime-texture-slot";
            Console.Error.WriteLine(
                $"ZORAH_MATERIAL_GRAPH_UNMAPPED object={objectPath} input={input} " +
                $"texture={texture.Reference} output_index={texture.OutputIndex} reason={reason}"
            );
        }
        return new MaterialGraphDefaults([.. scalars], [.. textures]);
    }

    // Bevy multiplies a metallic-roughness map's green channel into
    // perceptual_roughness, and UE's lerp(0, B, channel) is that same multiply,
    // so a roughness input that only rescales the packed channel survives as a
    // scalar and leaves the map itself untouched. UE serializes a property only
    // when it differs from the class default, so an absent ConstA is the engine's
    // zero; an absent ConstB is its one, which is the no-op this returns null for.
    private static object? ReadGraphRoughnessScale(
        object? input,
        Dictionary<string, UObject> expressions
    )
    {
        var (name, _) = GraphEdge(input);
        if (name is null ||
            expressions.GetValueOrDefault(name) is not { } lerp ||
            !lerp.ExportType.Equals("MaterialExpressionLinearInterpolate", StringComparison.Ordinal) ||
            GraphEdge(GetTaggedValue(lerp, "A")).Expression is not null ||
            GraphEdge(GetTaggedValue(lerp, "B")).Expression is not null ||
            GetTaggedValue(lerp, "ConstA") is not null)
        {
            return null;
        }
        return JsonScalar(GetTaggedValue(lerp, "ConstB"));
    }

    /// The expression an FExpressionInput reads, and the output index it takes
    /// off it (0 = RGB, 1..4 = the individual RGBA channels of a texture sample).
    private static (string? Expression, int OutputIndex) GraphEdge(object? input)
    {
        var expressionInput = StructValue(input);
        var name = JsonScalar(GetPublicMember(expressionInput, "ExpressionName"))?.ToString();
        return (
            string.IsNullOrEmpty(name) || string.Equals(name, "None", StringComparison.Ordinal)
                ? null
                : name,
            ToNullableInt(GetPublicMember(expressionInput, "OutputIndex")) ?? 0
        );
    }

    // Breadth-first, so a texture wired straight into a material input wins over
    // one further back behind a chain of maths. The reported output index is the
    // one on the edge that reaches the sample, which is how UE picks a single
    // channel out of a packed map.
    private static GraphTexture? ResolveGraphTexture(
        object? input,
        Dictionary<string, UObject> expressions
    )
    {
        var visited = new HashSet<string>(StringComparer.Ordinal);
        var pending = new Queue<(string Expression, int OutputIndex)>();
        Enqueue(input);
        while (pending.TryDequeue(out var edge))
        {
            if (!expressions.TryGetValue(edge.Expression, out var expression))
            {
                continue;
            }
            var texture = GetTaggedValue(expression, "Texture");
            if (texture is not null)
            {
                return new GraphTexture(
                    JsonScalar(GetTaggedValue(expression, "ParameterName"))?.ToString(),
                    JsonScalar(texture)?.ToString(),
                    edge.OutputIndex
                );
            }
            foreach (var property in expression.Properties)
            {
                Enqueue(property.Tag?.GenericValue);
            }
        }
        return null;

        void Enqueue(object? value)
        {
            var (name, outputIndex) = GraphEdge(value);
            if (name is not null && visited.Add(name))
            {
                pending.Enqueue((name, outputIndex));
            }
        }
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

    /// Unwrap an FScriptStruct to whichever native struct CUE4Parse read into it.
    private static object? StructValue(object? value) =>
        GetPublicMember(value, "StructType") ?? value;

    /// A parameter's ExpressionGUID as 32 uppercase hex digits.
    ///
    /// UE reconciles a material instance's overrides with its master by GUID and
    /// only falls back to the name, so an instance authored against an older
    /// revision keeps a name the master no longer has. An all-zero GUID is the
    /// unset value and is reported as absent.
    private static string? ReadGuid(object? value)
    {
        var text = new string(
            (StructValue(value)?.ToString() ?? "")
                .Where(char.IsAsciiLetterOrDigit)
                .ToArray()
        ).ToUpperInvariant();
        return text.Length == 0 || text.All(character => character == '0') ? null : text;
    }

    private static Dictionary<string, object?> ReadStructFields(object? value)
    {
        var properties = GetPublicMember(StructValue(value), "Properties") as IEnumerable;
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

    // A Niagara parameter store is a flat little-endian byte blob plus a sorted
    // (name, type, offset) index. FNiagaraVariableWithOffset serializes natively
    // rather than as tagged properties, so its members come off the CUE4Parse
    // struct itself instead of ReadStructFields.
    private static NiagaraParameterRecord[] ReadNiagaraParameterStore(object? store)
    {
        var fields = ReadStructFields(store);
        var data = ReadArrayValues(fields.GetValueOrDefault("ParameterData"))
            .OfType<byte>()
            .ToArray();
        var records = new List<NiagaraParameterRecord>();
        foreach (var entry in ReadArrayValues(
            fields.GetValueOrDefault("SortedParameterOffsets")
        ))
        {
            var variable = StructValue(entry);
            var type = JsonScalar(
                ReadStructFields(GetPublicMember(variable, "TypeDef"))
                    .GetValueOrDefault("ClassStructOrEnum")
            )?.ToString();
            records.Add(new NiagaraParameterRecord(
                Name: GetPublicMember(variable, "Name")?.ToString() ?? "unnamed",
                Type: type,
                Value: ReadNiagaraParameterValue(
                    data,
                    ToNullableInt(GetPublicMember(variable, "Offset")) ?? -1,
                    type
                )
            ));
        }
        return records.ToArray();
    }

    // `FNiagaraParameterStore`-shaped tags a `UNiagaraScript` carries besides
    // its rapid iteration parameters.
    private static readonly string[] NiagaraExecutionStoreNames =
    [
        "ScriptExecutionParamStore",
        "ScriptExecutionParamStoreCPU",
        "ExposedParameters",
        "OverrideParameters",
    ];

    // `FNiagaraVMExecutableData::Parameters` is the compiled script's external
    // parameter list: one `FNiagaraVariable` per constant the script reads,
    // each carrying its baked value in `VarData`. A module input left at its
    // module default has no rapid iteration entry but does appear here, so this
    // is the only place the default is recoverable.
    private static NiagaraParameterRecord[] ReadNiagaraCompiledParameters(object? executableData)
    {
        var parameters = ReadStructFields(
            ReadStructFields(executableData).GetValueOrDefault("Parameters")
        );
        var records = new List<NiagaraParameterRecord>();
        foreach (var entry in ReadArrayValues(parameters.GetValueOrDefault("Parameters")))
        {
            var variable = StructValue(entry);
            var type = JsonScalar(
                ReadStructFields(GetPublicMember(variable, "TypeDef"))
                    .GetValueOrDefault("ClassStructOrEnum")
            )?.ToString();
            var data = (GetPublicMember(variable, "VarData") as IEnumerable)?
                .Cast<object>()
                .OfType<byte>()
                .ToArray() ?? [];
            records.Add(new NiagaraParameterRecord(
                Name: GetPublicMember(variable, "Name")?.ToString() ?? "unnamed",
                Type: type,
                Value: ReadNiagaraParameterValue(data, 0, type)
            ));
        }
        return records.ToArray();
    }

    // Component counts for the Niagara types this project uses. An unlisted type
    // (a data interface, an enum, a struct) reads back null rather than a guess,
    // because a wrong width would silently shift every later parameter.
    private static readonly Dictionary<string, int> NiagaraParameterFloatCounts =
        new(StringComparer.Ordinal)
        {
            ["/Script/Niagara.NiagaraFloat"] = 1,
            ["/Script/CoreUObject.Vector2D"] = 2,
            ["/Script/CoreUObject.Vector2f"] = 2,
            ["/Script/Niagara.NiagaraPosition"] = 3,
            ["/Script/CoreUObject.Vector"] = 3,
            ["/Script/CoreUObject.Vector3f"] = 3,
            ["/Script/CoreUObject.Vector4"] = 4,
            ["/Script/CoreUObject.Vector4f"] = 4,
            ["/Script/CoreUObject.Quat"] = 4,
            ["/Script/CoreUObject.LinearColor"] = 4,
        };

    private static object? ReadNiagaraParameterValue(byte[] data, int offset, string? type)
    {
        if (offset < 0 || type is null)
        {
            return null;
        }
        // Niagara stores bools as int32 with -1 for true.
        if (type is "/Script/Niagara.NiagaraInt32" or "/Script/Niagara.NiagaraBool")
        {
            return offset + 4 <= data.Length
                ? BitConverter.ToInt32(data, offset)
                : null;
        }
        if (!NiagaraParameterFloatCounts.TryGetValue(type, out var count) ||
            offset + (4 * count) > data.Length)
        {
            return null;
        }
        var components = new float[count];
        for (var index = 0; index < count; index++)
        {
            components[index] = BitConverter.ToSingle(data, offset + (4 * index));
        }
        return count == 1 ? components[0] : components;
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
                var attach = AttachParentOutsideActor(rootComponent);
                Console.WriteLine(
                    $"ZORAH_ACTOR_ATTACH actor={obj.Name} " +
                    $"parent_actor={attach?.Actor ?? "None"} " +
                    $"parent_component={attach?.Component ?? "None"}"
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
            // Niagara emitters, renderers and parameter stores are plain tagged
            // UObjects to CUE4Parse, with no typed wrapper to read them through,
            // so dump every tag and let the caller pick the ones it needs.
            if (obj.ExportType.StartsWith("Niagara", StringComparison.Ordinal))
            {
                foreach (var property in obj.Properties)
                {
                    Console.WriteLine(
                        $"ZORAH_NIAGARA_PROPERTY object={obj.Name} type={obj.ExportType} " +
                        $"name={property.Name.Text} value=" +
                        JsonSerializer.Serialize(
                            JsonScalar(property.Tag?.GenericValue),
                            JsonOptions
                        )
                    );
                }
                foreach (var parameter in ReadNiagaraParameterStore(
                    GetTaggedValue(obj, "RapidIterationParameters")
                ))
                {
                    Console.WriteLine(
                        $"ZORAH_NIAGARA_PARAMETER object={obj.Name} " +
                        $"name={parameter.Name} type={parameter.Type ?? "None"} " +
                        $"value={JsonSerializer.Serialize(parameter.Value, JsonOptions)}"
                    );
                }
                // A module input left at its module default has no rapid
                // iteration entry; the compiler bakes the default into the
                // script's execution store instead, so read that too.
                foreach (var storeName in NiagaraExecutionStoreNames)
                {
                    foreach (var parameter in ReadNiagaraParameterStore(
                        GetTaggedValue(obj, storeName)
                    ))
                    {
                        Console.WriteLine(
                            $"ZORAH_NIAGARA_STORE_PARAMETER object={obj.Name} " +
                            $"store={storeName} name={parameter.Name} " +
                            $"type={parameter.Type ?? "None"} " +
                            $"value={JsonSerializer.Serialize(parameter.Value, JsonOptions)}"
                        );
                    }
                }
                foreach (var parameter in ReadNiagaraCompiledParameters(
                    GetTaggedValue(obj, "CachedScriptVM")
                ))
                {
                    Console.WriteLine(
                        $"ZORAH_NIAGARA_COMPILED_PARAMETER object={obj.Name} " +
                        $"name={parameter.Name} type={parameter.Type ?? "None"} " +
                        $"value={JsonSerializer.Serialize(parameter.Value, JsonOptions)}"
                    );
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
            if (obj is UChildActorComponent childActorComponent)
            {
                Console.WriteLine(
                    "ZORAH_CHILD_ACTOR_COMPONENT " +
                    $"child_actor={ChildActor(childActorComponent)?.Name ?? "None"}"
                );
                foreach (var record in ConvertChildActorComponents(
                    provider,
                    objects,
                    InspectArchetypes(provider, obj, objects),
                    [obj]
                ))
                {
                    Console.WriteLine(
                        $"ZORAH_CHILD_ACTOR_MESH name={record.Name} " +
                        $"mesh={record.Mesh ?? "None"} " +
                        $"translation={Format(record.Transform.Translation)} " +
                        $"rotation={Format(record.Transform.Rotation)} " +
                        $"scale={Format(record.Transform.Scale)}"
                    );
                }
            }
            if (obj.ExportType == "DecalComponent")
            {
                Console.WriteLine(
                    "ZORAH_DECAL_RECORD " +
                    JsonSerializer.Serialize(
                        ConvertDecalComponent(obj, InspectArchetypes(provider, obj, objects)),
                        JsonOptions
                    )
                );
                foreach (var property in obj.Properties)
                {
                    Console.WriteLine(
                        $"ZORAH_DECAL_PROPERTY name={property.Name.Text} " +
                        $"value={Describe(property.Tag?.GenericValue)}"
                    );
                }
            }
            // A WorldDataLayers actor, its DataLayerInstance subobjects and the
            // UDataLayerAsset packages they point at are plain tagged UObjects,
            // so dump every tag and let the caller read the layer definitions.
            // Any other actor gets only its layer membership dumped.
            var dataLayerObject = obj.ExportType.Contains("DataLayer", StringComparison.Ordinal);
            foreach (var property in obj.Properties)
            {
                if (!dataLayerObject &&
                    property.Name.Text is not ("DataLayerAssets" or "DataLayers"))
                {
                    continue;
                }
                Console.WriteLine(
                    $"ZORAH_DATA_LAYER_PROPERTY object={obj.Name} type={obj.ExportType} " +
                    $"name={property.Name.Text} value={Describe(property.Tag?.GenericValue)}"
                );
                DumpNested(
                    $"data_layer.{obj.Name}.{property.Name.Text}",
                    property.Tag?.GenericValue,
                    3,
                    new HashSet<object>(ReferenceEqualityComparer.Instance)
                );
            }
            if (IsExternalActorRoot(obj))
            {
                Console.WriteLine(
                    $"ZORAH_ACTOR_DATA_LAYERS actor={obj.Name} " +
                    $"layers={string.Join(",", ReadActorDataLayers(obj))}"
                );
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
            actor is null ? [] : OwnedComponents(actor, packageObjects),
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
        var referencedDecalMaterials = new HashSet<string>(StringComparer.Ordinal);
        var referencedNiagaraMaterials = new HashSet<string>(StringComparer.Ordinal);
        var missingNiagaraAssets = new HashSet<string>(StringComparer.Ordinal);
        var pendingAttachments = new Dictionary<int, PendingAttachment>();
        var declaredDataLayers = new Dictionary<string, DataLayerRecord>(StringComparer.Ordinal);
        var dataLayerActorCounts = new Dictionary<string, int>(StringComparer.Ordinal);
        var dataLayerAssets = new Dictionary<string, string>(StringComparer.Ordinal);
        // Placed instances of the same system are common - ThroneRoom has 6,421
        // candles fed by one - and reading a system means loading and walking a
        // package with hundreds of exports.
        var niagaraSystems = new Dictionary<string, NiagaraMeshRendererRecord[]>(
            StringComparer.Ordinal
        );
        var packagesWithoutActors = 0;

        void RequestNiagaraAsset(string? path, HashSet<string> requested)
        {
            if (path is not null)
            {
                (provider.Files.ContainsKey(ObjectPathToPackageKey(path))
                    ? requested
                    : missingNiagaraAssets).Add(path);
            }
        }

        for (var packageIndex = 0; packageIndex < packagePaths.Length; packageIndex++)
        {
            var packagePath = packagePaths[packageIndex];
            try
            {
                var objects = provider.LoadPackage(packagePath).GetExports().ToArray();
                var childActorNames = objects
                    .OfType<UChildActorComponent>()
                    .Select(ChildActor)
                    .OfType<UObject>()
                    .Select(child => child.Name)
                    .ToHashSet(StringComparer.Ordinal);
                var packageActors = objects
                    .Where(IsExternalActorRoot)
                    .Where(actor => !childActorNames.Contains(actor.Name))
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
                    var owned = OwnedComponents(actor, objects);
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
                        .Concat(ConvertChildActorComponents(provider, objects, archetypes, owned))
                        .ToArray();

                    var niagara = owned
                        .OfType<UNiagaraComponent>()
                        .OrderBy(component => component.Name, StringComparer.Ordinal)
                        .Select(component => ConvertNiagaraComponent(
                            provider,
                            component,
                            archetypes,
                            niagaraSystems
                        ))
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

                    var decals = owned
                        .Where(component => component.ExportType == "DecalComponent")
                        .OrderBy(component => component.Name, StringComparer.Ordinal)
                        .Select(component => ConvertDecalComponent(component, archetypes))
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
                    foreach (var decal in decals)
                    {
                        if (decal.Material is not null)
                        {
                            referencedDecalMaterials.Add(decal.Material);
                        }
                    }
                    // A mesh renderer's mesh reaches the scene through the
                    // particle system rather than through a component property,
                    // so it has to be collected here or geometry conversion
                    // never sees it. Renderers reach for /Engine primitives
                    // (BasicShapes/Sphere and friends) that the sample download
                    // does not ship, so only assets the provider actually holds
                    // are requested.
                    foreach (var renderer in niagara
                        .SelectMany(component => component.MeshRenderers))
                    {
                        foreach (var mesh in renderer.Meshes)
                        {
                            RequestNiagaraAsset(mesh.Mesh, referencedMeshes);
                        }
                        foreach (var material in renderer.OverrideMaterials)
                        {
                            RequestNiagaraAsset(material, referencedNiagaraMaterials);
                        }
                    }

                    var attachParent = AttachParentOutsideActor(rootComponent);
                    if (rootComponent is not null && attachParent is not null)
                    {
                        pendingAttachments[actors.Count] = new PendingAttachment(
                            ParentActor: attachParent.Value.Actor,
                            ParentComponent: attachParent.Value.Component,
                            Relative: ConvertObjectTransform(rootComponent),
                            AbsoluteLocation: rootComponent.GetOrDefault(
                                "bAbsoluteLocation",
                                false
                            ),
                            AbsoluteRotation: rootComponent.GetOrDefault(
                                "bAbsoluteRotation",
                                false
                            ),
                            AbsoluteScale: rootComponent.GetOrDefault("bAbsoluteScale", false)
                        );
                    }

                    var dataLayerPaths = ReadActorDataLayers(actor);
                    foreach (var path in dataLayerPaths)
                    {
                        var layer = ShortObjectName(path);
                        Increment(dataLayerActorCounts, layer);
                        if (path.Contains('/'))
                        {
                            dataLayerAssets.TryAdd(layer, path);
                        }
                    }
                    if (actor.ExportType == "WorldDataLayers")
                    {
                        foreach (var (name, record) in ReadWorldDataLayers(
                            provider,
                            actor,
                            objects
                        ))
                        {
                            declaredDataLayers[name] = record;
                        }
                    }

                    actors.Add(new ActorRecord(
                        Package: packagePath,
                        Name: actor.Name,
                        Label: actor.GetOrDefault<string?>("ActorLabel", null),
                        Type: actor.ExportType,
                        Class: actor.Class?.GetPathName(),
                        Transform: ConvertObjectTransform(rootComponent),
                        AttachParent: attachParent?.Actor,
                        Hidden: actor.GetOrDefault("bHidden", false),
                        DataLayers: dataLayerPaths.Length == 0
                            ? null
                            : dataLayerPaths.Select(ShortObjectName).ToArray(),
                        Components: components,
                        Niagara: niagara,
                        Lights: lights,
                        Decals: decals,
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

        ResolveAttachedActors(provider, level, actors, pendingAttachments);

        // A layer an actor claims but WorldDataLayers never declares still gets
        // an entry so the runtime can see it; its state stays unknown.
        var dataLayers = declaredDataLayers.Keys
            .Concat(dataLayerActorCounts.Keys)
            .Distinct(StringComparer.Ordinal)
            .Order(StringComparer.Ordinal)
            .Select(name => declaredDataLayers.TryGetValue(name, out var declared)
                ? declared
                : new DataLayerRecord(
                    Name: name,
                    Asset: dataLayerAssets.GetValueOrDefault(name),
                    Type: ReadDataLayerType(provider, dataLayerAssets.GetValueOrDefault(name)),
                    InitialRuntimeState: null,
                    InitiallyVisible: null,
                    InitiallyLoadedInEditor: null
                ))
            .ToArray();

        var manifest = new SceneManifest(
            Format: "zorah-scene-manifest-v6",
            EngineVersion: "5.4",
            Level: level,
            SourceMap: $"Levels/{level}.umap",
            ExternalActorPrefix: prefix,
            ActorPackageCount: packagePaths.Length,
            PackagesWithoutActors: packagesWithoutActors,
            DataLayers: dataLayers,
            ActorTypeCounts: actorTypes.OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(),
            StaticMeshComponentTypeCounts: componentTypes
                .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                .ToDictionary(),
            UnresolvedStaticMeshComponents: actors
                .SelectMany(actor => actor.Components)
                .Count(component => component.Mesh is null),
            ReferencedMeshes: referencedMeshes.Order(StringComparer.Ordinal).ToArray(),
            DecalComponents: actors.Sum(actor => actor.Decals.Length),
            ReferencedDecalMaterials: referencedDecalMaterials
                .Order(StringComparer.Ordinal)
                .ToArray(),
            NiagaraComponents: actors.Sum(actor => actor.Niagara.Length),
            ReferencedNiagaraMaterials: referencedNiagaraMaterials
                .Order(StringComparer.Ordinal)
                .ToArray(),
            MissingNiagaraAssets: missingNiagaraAssets.Order(StringComparer.Ordinal).ToArray(),
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
            $"ZORAH_LEVEL_ATTACHED_ACTORS level={level} count={pendingAttachments.Count}"
        );
        Console.WriteLine(
            $"ZORAH_LEVEL_DATA_LAYERS level={level} layers=" +
            string.Join(",", manifest.DataLayers.Select(layer =>
                $"{layer.Name}:{layer.InitialRuntimeState ?? "None"}:" +
                dataLayerActorCounts.GetValueOrDefault(layer.Name)
            ))
        );
        Console.WriteLine(
            $"ZORAH_LEVEL_DONE level={level} actors={manifest.Actors.Length} " +
            $"lights={manifest.Actors.Sum(actor => actor.Lights.Length)} " +
            $"meshes={manifest.ReferencedMeshes.Length} failures={manifest.Failures.Length} " +
            $"unresolved_mesh_components={manifest.UnresolvedStaticMeshComponents} " +
            $"niagara={manifest.NiagaraComponents} " +
            $"niagara_missing={manifest.MissingNiagaraAssets.Length} " +
            $"light_functions_without_mean={manifest.Actors
                .SelectMany(actor => actor.Lights)
                .Count(light => light.LightFunctionMaterial is not null
                    && light.LightFunctionMean is null)} " +
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

    private static UObject[] OwnedComponents(UObject actor, UObject[] packageObjects) =>
        packageObjects.Where(obj => FindOwningActor(obj)?.Name == actor.Name).ToArray();

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

        var owned = OwnedComponents(actor, packageObjects);
        return owned.FirstOrDefault(obj => obj.Name is "DefaultSceneRoot" or "DefaultSceneRoot_0")
            ?? owned.FirstOrDefault(obj => obj.ExportType.EndsWith(
                "RootComponent",
                StringComparison.Ordinal
            ))
            ?? (owned.Length == 1 ? owned[0] : null);
    }

    /// The actor and component an actor root is attached to outside its own package.
    ///
    /// Such an AttachParent is an import that resolves through the persistent
    /// level - /Game/Levels/Level.Level:PersistentLevel.Actor.Component - so the
    /// parent's own external actor package is not named here. Attachments inside
    /// the actor are exports and stay with ComponentArchetypes.
    private static (string Actor, string Component)? AttachParentOutsideActor(
        UObject? component
    )
    {
        if (component is null ||
            GetTaggedValue(component, "AttachParent") is not FPackageIndex index ||
            !index.IsImport)
        {
            return null;
        }
        var resolved = index.ResolvedObject;
        var actor = resolved?.Outer?.Name.Text;
        return actor is null ? null : (actor, resolved!.Name.Text);
    }

    private sealed record PendingAttachment(
        string ParentActor,
        string ParentComponent,
        TransformRecord Relative,
        bool AbsoluteLocation,
        bool AbsoluteRotation,
        bool AbsoluteScale
    );

    /// Rewrite attached actors' transforms from parent-relative to world space.
    ///
    /// An actor root attached to another actor's component serializes its
    /// transform relative to that component, but the manifest stores actor
    /// transforms in world space. The parent sits in its own external actor
    /// package, so this only resolves once the whole level is read. A parent may
    /// itself be attached, hence the recursion. Anything unresolvable keeps the
    /// relative transform and is reported rather than throwing: one misplaced
    /// actor beats no manifest.
    private static void ResolveAttachedActors(
        DefaultFileProvider provider,
        string level,
        List<ActorRecord> actors,
        Dictionary<int, PendingAttachment> pending
    )
    {
        var actorIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (var index = 0; index < actors.Count; index++)
        {
            actorIndex.TryAdd(actors[index].Name, index);
        }
        var offsets = new Dictionary<string, TransformRecord?>(StringComparer.Ordinal);
        var world = new Dictionary<int, TransformRecord>();
        var resolving = new HashSet<int>();

        TransformRecord? ComponentOffset(ActorRecord parent, string component)
        {
            var key = string.Join('\0', parent.Package, parent.Name, component);
            if (offsets.TryGetValue(key, out var cached))
            {
                return cached;
            }
            TransformRecord? offset = null;
            try
            {
                var objects = provider.LoadPackage(parent.Package).GetExports().ToArray();
                if (objects.FirstOrDefault(obj => obj.Name == parent.Name) is UObject owner)
                {
                    var owned = OwnedComponents(owner, objects);
                    if (owned.FirstOrDefault(obj => obj.Name == component) is UObject target)
                    {
                        offset = new ComponentArchetypes(
                            owner,
                            FindRootComponent(owner, objects),
                            owned,
                            LoadBlueprintComponentTemplates(provider, owner.Class?.GetPathName())
                        ).TransformRelativeToActor(target);
                    }
                }
            }
            catch (Exception error)
            {
                Console.Error.WriteLine(
                    $"ZORAH_ATTACH_UNRESOLVED level={level} parent={parent.Name} " +
                    $"component={component} reason={error.GetType().Name}"
                );
            }
            offsets[key] = offset;
            return offset;
        }

        TransformRecord Resolve(int index)
        {
            if (world.TryGetValue(index, out var cached))
            {
                return cached;
            }
            if (!pending.TryGetValue(index, out var attachment))
            {
                world[index] = actors[index].Transform;
                return actors[index].Transform;
            }
            var transform = attachment.Relative;
            if (!resolving.Add(index))
            {
                Console.Error.WriteLine(
                    $"ZORAH_ATTACH_UNRESOLVED level={level} actor={actors[index].Name} " +
                    $"parent={attachment.ParentActor} reason=cycle"
                );
                return transform;
            }
            try
            {
                if (!actorIndex.TryGetValue(attachment.ParentActor, out var parent))
                {
                    Console.Error.WriteLine(
                        $"ZORAH_ATTACH_UNRESOLVED level={level} actor={actors[index].Name} " +
                        $"parent={attachment.ParentActor} reason=missing-actor"
                    );
                }
                else if (ComponentOffset(actors[parent], attachment.ParentComponent)
                    is not TransformRecord offset)
                {
                    Console.Error.WriteLine(
                        $"ZORAH_ATTACH_UNRESOLVED level={level} actor={actors[index].Name} " +
                        $"parent={attachment.ParentActor} " +
                        $"component={attachment.ParentComponent} reason=missing-component"
                    );
                }
                else
                {
                    transform = ComposeTransforms(
                        ComposeTransforms(Resolve(parent), offset),
                        attachment.Relative
                    );
                    // An absolute flag keeps the child's own value in world space
                    // rather than composing it onto the parent.
                    if (attachment.AbsoluteLocation)
                    {
                        transform = transform with
                        {
                            Translation = attachment.Relative.Translation,
                        };
                    }
                    if (attachment.AbsoluteRotation)
                    {
                        transform = transform with { Rotation = attachment.Relative.Rotation };
                    }
                    if (attachment.AbsoluteScale)
                    {
                        transform = transform with { Scale = attachment.Relative.Scale };
                    }
                }
            }
            finally
            {
                resolving.Remove(index);
            }
            world[index] = transform;
            return transform;
        }

        foreach (var index in pending.Keys.Order())
        {
            actors[index] = actors[index] with { Transform = Resolve(index) };
        }
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

    // The blueprint's component keeps a ChildActorTemplate; a placed instance
    // serializes the actor it spawned from that template as ChildActor instead.
    private static UObject? ChildActor(UChildActorComponent component) =>
        (GetTaggedValue(component, "ChildActor") as FPackageIndex)?.Load();

    // UChildActorComponent spawns its child at the component's world transform
    // and attaches the spawned root to the component, so the child's own root
    // transform never applies. The level saves that child as a package-level
    // export sitting beside its parent, which is why its meshes have to be
    // re-parented here instead of being emitted as their own actor.
    private static StaticMeshComponentRecord[] ConvertChildActorComponents(
        DefaultFileProvider provider,
        UObject[] packageObjects,
        ComponentArchetypes archetypes,
        UObject[] ownedComponents
    )
    {
        var records = new List<StaticMeshComponentRecord>();
        foreach (var component in ownedComponents
            .OfType<UChildActorComponent>()
            .OrderBy(component => component.Name, StringComparer.Ordinal))
        {
            var childActor = ChildActor(component);
            if (childActor is null)
            {
                continue;
            }
            var childOwned = OwnedComponents(childActor, packageObjects);
            var childArchetypes = new ComponentArchetypes(
                childActor,
                FindRootComponent(childActor, packageObjects),
                childOwned,
                LoadBlueprintComponentTemplates(provider, childActor.Class?.GetPathName())
            );
            var childTransform = archetypes.TransformRelativeToActor(component);
            records.AddRange(childOwned
                .OfType<UStaticMeshComponent>()
                .OrderBy(mesh => mesh.Name, StringComparer.Ordinal)
                .Select(mesh => ConvertComponent(provider, mesh, childArchetypes))
                .Select(record => record with
                {
                    Name = $"{component.Name}.{record.Name}",
                    Transform = ComposeTransforms(childTransform, record.Transform),
                }));
        }
        return records.ToArray();
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

    /// <summary>
    /// Spatial and temporal mean of a light function material's emissive
    /// output, by material path.
    /// </summary>
    /// <remarks>
    /// UE multiplies a light per-pixel by its light function material, so the
    /// light's effective average output is its flux times this mean. A uniform
    /// emissive proxy carries no spatial modulation, and the mean is the only
    /// part of the pattern it can express.
    ///
    /// LF_LIghtCaustics_01 overrides nothing on its parent, whose EmissiveColor
    /// is 2*C(uv_a, t) * 3*C(uv_b, t) for C = T_Caustics_01_MSK (4096^2,
    /// TSF_G16, SRGB=false, so already linear) read through Panners at
    /// (0.003, -0.02) and (0.005, 0.03) tiles/s. E[C] = 0.037083 over all
    /// 16.7M texels. The two samples decorrelate: the FFT autocorrelation
    /// averaged along the real relative-pan trajectory is 0.0013759 against
    /// E[C]^2 = 0.0013751, so E[6 C_a C_b] = 6 E[C]^2 = 0.008255. UE writes
    /// light functions through an 8-bit UNORM attenuation buffer, an implicit
    /// saturate retaining 98.24%, which lands at 0.00811.
    /// </remarks>
    private static readonly Dictionary<string, double> LightFunctionMeans =
        new(StringComparer.OrdinalIgnoreCase)
        {
            ["/Game/VFX/NebulaOrb/LF_LIghtCaustics_01_Inst.LF_LIghtCaustics_01_Inst"] = 0.0081,
        };

    private static double? MeasuredLightFunctionMean(string? material) =>
        material is not null && LightFunctionMeans.TryGetValue(material, out var mean)
            ? mean
            : null;

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
        var lightFunctionMaterial = ReadReference(chain, "LightFunctionMaterial");
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
            LightFunctionMaterial: lightFunctionMaterial,
            LightFunctionMean: MeasuredLightFunctionMean(lightFunctionMaterial),
            // ULightComponent CDO defaults. Nothing in the sample overrides
            // them, so they are exported to make that visible rather than
            // assumed: at a 1 km fade distance in a 28 m room the distance fade
            // never runs, which is what keeps DisabledBrightness out of the
            // light function's effective output.
            LightFunctionScale: ReadVector(
                chain,
                "LightFunctionScale",
                new FVector(1024.0f, 1024.0f, 1024.0f)
            ),
            LightFunctionFadeDistance: ReadDouble(chain, "LightFunctionFadeDistance", 100000.0),
            DisabledBrightness: ReadDouble(chain, "DisabledBrightness", 0.5),
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

    // UDecalComponent's class default, in centimetres. Every DecalActor in the
    // three levels serializes DecalMaterial and its relative transform and
    // nothing else, so the box itself is delta-elided against this. Confirmed
    // against the ActorMetaData each external actor carries, which records the
    // actor's world bounds: for DecalActor_UAID_A8A159F1FA82944602_1162180937
    // (MI_Decal_SootStain_A7) |R| * (DecalSize * RelativeScale3D) reproduces the
    // stored extent (42.24740, 91.89775, 153.32378) to seven figures, which no
    // other size does.
    private static readonly FVector DefaultDecalSize = new(128.0f, 256.0f, 256.0f);

    private static DecalComponentRecord ConvertDecalComponent(
        UObject component,
        ComponentArchetypes archetypes
    )
    {
        var chain = archetypes.Chain(component);
        return new DecalComponentRecord(
            Name: component.Name,
            Type: component.ExportType,
            Transform: archetypes.TransformRelativeToActor(component),
            Visible: ReadBool(chain, "bVisible", true),
            HiddenInGame: ReadBool(chain, "bHiddenInGame", false),
            Material: ReadReference(chain, "DecalMaterial"),
            Size: ReadVector(chain, "DecalSize", DefaultDecalSize),
            SortOrder: ToNullableInt(ReadTaggedValue(chain, "SortOrder")) ?? 0,
            FadeScreenSize: ReadDouble(chain, "FadeScreenSize", 0.01),
            FadeStartDelay: ReadDouble(chain, "FadeStartDelay", 0.0),
            FadeDuration: ReadDouble(chain, "FadeDuration", 0.0),
            FadeInStartDelay: ReadDouble(chain, "FadeInStartDelay", 0.0),
            FadeInDuration: ReadDouble(chain, "FadeInDuration", 0.0)
        );
    }

    private static NiagaraComponentRecord ConvertNiagaraComponent(
        DefaultFileProvider provider,
        UNiagaraComponent component,
        ComponentArchetypes archetypes,
        Dictionary<string, NiagaraMeshRendererRecord[]> systemCache
    )
    {
        var chain = archetypes.Chain(component);
        // A blueprint's SimpleConstructionScript node keeps Asset and the
        // relative transform; the placed instance serializes neither, which is
        // why this has to read through the archetype chain the same way
        // ConvertComponent does for StaticMesh.
        var asset = ReadReference(chain, "Asset");
        NiagaraMeshRendererRecord[] renderers = [];
        if (asset is not null)
        {
            if (!systemCache.TryGetValue(asset, out var cached))
            {
                cached = ReadNiagaraMeshRenderers(provider, asset);
                systemCache[asset] = cached;
            }
            renderers = cached;
        }
        return new NiagaraComponentRecord(
            Name: component.Name,
            Type: component.ExportType,
            Asset: asset,
            Transform: archetypes.TransformRelativeToActor(component),
            Visible: ReadBool(chain, "bVisible", true),
            HiddenInGame: ReadBool(chain, "bHiddenInGame", false),
            AutoActivate: ReadBool(chain, "bAutoActivate", true),
            VisibleInRayTracing: ReadBool(chain, "bVisibleInRayTracing", true),
            // Unlike a UStaticMeshComponent, a particle system does not cast by
            // default: the butterfly component serializes CastShadow=true, which
            // it would not if true were the class default.
            CastShadow: ReadBool(chain, "CastShadow", false),
            OverrideParameters: ReadNiagaraParameterStore(
                ReadTaggedValue(chain, "OverrideParameters")
            ),
            MeshRenderers: renderers
        );
    }

    // Mesh renderers live on the emitters inside the system asset, not on the
    // component, so the meshes and their material overrides are only reachable
    // by opening that package. A renderer whose bIsEnabled is false is still
    // reported: the emitter that owns it may be enabled, and dropping it here
    // would hide the reason a mesh never appears.
    private static NiagaraMeshRendererRecord[] ReadNiagaraMeshRenderers(
        DefaultFileProvider provider,
        string systemPath
    )
    {
        var packagePath = ObjectPathToPackageKey(systemPath);
        if (!provider.Files.ContainsKey(packagePath))
        {
            return [];
        }
        var records = new List<NiagaraMeshRendererRecord>();
        foreach (var renderer in provider.LoadPackage(packagePath).GetExports()
            .Where(export => export.ExportType == "NiagaraMeshRendererProperties"))
        {
            var meshes = ReadArrayValues(GetTaggedValue(renderer, "Meshes"))
                .Select(ReadStructFields)
                .Select(mesh => new NiagaraRendererMeshRecord(
                    Mesh: PackageReferencePath(mesh.GetValueOrDefault("Mesh")),
                    Scale: ReadVec3Record(mesh.GetValueOrDefault("Scale"), 1.0),
                    PivotOffset: ReadVec3Record(mesh.GetValueOrDefault("PivotOffset"), 0.0)
                ))
                .ToArray();
            records.Add(new NiagaraMeshRendererRecord(
                // The renderer's outer is its emitter; the system names emitters
                // by that outer, so this is the handle to match against.
                Emitter: renderer.Outer?.Name.Text ?? "None",
                Enabled: ToBool(GetTaggedValue(renderer, "bIsEnabled"), true),
                Meshes: meshes,
                OverrideMaterials: ToBool(
                    GetTaggedValue(renderer, "bOverrideMaterials"),
                    false
                )
                    ? ReadArrayValues(GetTaggedValue(renderer, "OverrideMaterials"))
                        .Select(ReadStructFields)
                        .Select(entry =>
                            PackageReferencePath(entry.GetValueOrDefault("ExplicitMat")))
                        .OfType<string>()
                        .ToArray()
                    : []
            ));
        }
        return records.OrderBy(record => record.Emitter, StringComparer.Ordinal).ToArray();
    }

    private static Vec3Record ReadVec3Record(object? value, double defaultComponent)
    {
        var fields = ReadStructFields(value);
        return new Vec3Record(
            ToNullableDouble(fields.GetValueOrDefault("X")) ?? defaultComponent,
            ToNullableDouble(fields.GetValueOrDefault("Y")) ?? defaultComponent,
            ToNullableDouble(fields.GetValueOrDefault("Z")) ?? defaultComponent
        );
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

    private static Vec3Record ReadVector(UObject[] chain, string name, FVector defaultValue)
    {
        var value = ReadStruct(chain, name, defaultValue);
        return new Vec3Record(value.X, value.Y, value.Z);
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
    DataLayerRecord[] DataLayers,
    Dictionary<string, int> ActorTypeCounts,
    Dictionary<string, int> StaticMeshComponentTypeCounts,
    int UnresolvedStaticMeshComponents,
    string[] ReferencedMeshes,
    int DecalComponents,
    string[] ReferencedDecalMaterials,
    int NiagaraComponents,
    string[] ReferencedNiagaraMaterials,
    // Meshes and materials a Niagara renderer asks for that are not in the
    // sample download - all of them /Engine primitives.
    string[] MissingNiagaraAssets,
    ActorRecord[] Actors,
    FailureRecord[] Failures
);

sealed record DataLayerRecord(
    string Name,
    // Always emitted, null included: the runtime distinguishes a layer with no
    // known definition from one that simply defaults.
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] string? Asset,
    string Type,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] string? InitialRuntimeState,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] bool? InitiallyVisible,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.Never)] bool? InitiallyLoadedInEditor
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
    // The actor this one's root is attached to, when that is another actor.
    string? AttachParent,
    bool Hidden,
    string[]? DataLayers,
    StaticMeshComponentRecord[] Components,
    NiagaraComponentRecord[] Niagara,
    LightComponentRecord[] Lights,
    DecalComponentRecord[] Decals,
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
    double? LightFunctionMean,
    Vec3Record LightFunctionScale,
    double LightFunctionFadeDistance,
    double DisabledBrightness,
    bool RealTimeCapture
);

/// <summary>A placed UDecalComponent: the box UE projects DecalMaterial through.</summary>
/// <remarks>
/// Size is the box's half-extent in centimetres along the component's own axes,
/// with X the projection depth, so the projected rectangle measures
/// 2*Size.Y by 2*Size.Z once the component scale is applied.
/// </remarks>
sealed record DecalComponentRecord(
    string Name,
    string Type,
    TransformRecord Transform,
    bool Visible,
    bool HiddenInGame,
    string? Material,
    Vec3Record Size,
    int SortOrder,
    double FadeScreenSize,
    double FadeStartDelay,
    double FadeDuration,
    double FadeInStartDelay,
    double FadeInDuration
);

/// A placed UNiagaraComponent.
///
/// Nothing downstream simulates particles. The record exists so the meshes a
/// system renders reach geometry conversion, and so a consumer can place a
/// system's static output at the right transform.
sealed record NiagaraComponentRecord(
    string Name,
    string Type,
    string? Asset,
    TransformRecord Transform,
    bool Visible,
    bool HiddenInGame,
    bool AutoActivate,
    bool VisibleInRayTracing,
    bool CastShadow,
    NiagaraParameterRecord[] OverrideParameters,
    NiagaraMeshRendererRecord[] MeshRenderers
);

sealed record NiagaraMeshRendererRecord(
    string Emitter,
    bool Enabled,
    NiagaraRendererMeshRecord[] Meshes,
    string[] OverrideMaterials
);

sealed record NiagaraRendererMeshRecord(
    string? Mesh,
    Vec3Record Scale,
    Vec3Record PivotOffset
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

/// ExpressionGuid is null for a parameter that has none to carry: a texture
/// reached by walking a base material's graph, and the diagnostic stand-ins for
/// materials whose package is missing.
sealed record MaterialParameterRecord(
    string Name,
    string? Association,
    int? Index,
    object? Value,
    string? ExpressionGuid = null
);

sealed record StaticSwitchParameterRecord(
    string Name,
    string? Association,
    int? Index,
    bool Value,
    bool Override,
    string? ExpressionGuid = null
);

sealed record MaterialLayerFunctions(string[] Layers, string[] Blends);

/// A texture reached by walking a base UMaterial's expression graph.
/// ParameterName is null when the sampling node is not a material parameter, and
/// OutputIndex is the sample output the reaching edge read (0 = RGB, 1..4 = RGBA).
sealed record GraphTexture(string? ParameterName, string? Reference, int OutputIndex);

sealed record MaterialGraphDefaults(
    MaterialParameterRecord[] Scalars,
    MaterialParameterRecord[] Textures
);

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

sealed record NiagaraParameterRecord(
    string Name,
    string? Type,
    object? Value
);

sealed record TextureBlockRecord(
    int BlockX,
    int BlockY,
    int Width,
    int Height,
    long PayloadOffset,
    long PayloadSize
);
