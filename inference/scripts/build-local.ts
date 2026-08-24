import { mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { IcnInstallationDeclaration } from "@magnitudedev/icn-protocol";
import { Schema } from "effect";

/**
 * Stages a development ICN installation.
 *
 * Much smaller than it was. The previous build produced a native inference engine: shared
 * libraries staged into `runtime/`, per-accelerator backend modules into `backends/`, a GGUF
 * planner bundle into `catalog/`, and a Cargo feature graph selecting CUDA, Metal, or Vulkan.
 * None of that exists now — inference happens in an EIM container, so there is one binary and
 * one acceleration story, and what the container can use is discovered at runtime by the Docker
 * preflight rather than chosen at compile time.
 *
 * The layout still matches what the client's development binary resolution expects: it derives
 * `bin/magnitude-icn` from the directory holding `installation.json`.
 */

const PROJECT_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const MANIFEST = resolve(PROJECT_ROOT, "inference/Cargo.toml");

const run = async (
  command: readonly string[],
  options: { readonly captureStdout?: boolean } = {}
): Promise<string> => {
  const child = Bun.spawn([...command], {
    cwd: PROJECT_ROOT,
    stdin: "ignore",
    stdout: options.captureStdout ? "pipe" : "inherit",
    stderr: "inherit",
  });
  const stdout = options.captureStdout
    ? await new Response(child.stdout).text()
    : "";
  const code = await child.exited;
  if (code !== 0) {
    throw new Error(
      `command failed with exit code ${code}: ${command.join(" ")}`
    );
  }
  return stdout;
};

const executableName = (): string =>
  process.platform === "win32" ? "magnitude-icn.exe" : "magnitude-icn";

export const buildLocalIcn = async (): Promise<{
  readonly installationPath: string;
  readonly binaryPath: string;
}> => {
  console.log("[dev] Building ICN...");
  await run([
    "cargo",
    "build",
    "--manifest-path",
    MANIFEST,
    "--package",
    "icn-server",
  ]);

  const built = resolve(
    PROJECT_ROOT,
    "inference/target/debug",
    executableName()
  );
  // The declaration must describe the binary that will actually run, so read it from the binary
  // rather than recomputing the pins here and risking a mismatch.
  const identity = JSON.parse(await run([built, "version", "--json"], {
    captureStdout: true,
  })) as {
    readonly native_build: string;
    readonly backend_module_abi: string;
  };

  const target = resolve(PROJECT_ROOT, "inference/target");
  const staging = await mkdtemp(resolve(target, ".development-"));
  const destination = resolve(target, "development");
  try {
    // `runtime` stays as an empty directory: there are no native libraries to stage, but the
    // client still points a loader path at it and an existing directory keeps that harmless.
    for (const directory of ["bin", "runtime"]) {
      await mkdir(resolve(staging, directory), { recursive: true, mode: 0o700 });
    }
    await Bun.write(
      Bun.file(resolve(staging, "bin", executableName())),
      Bun.file(built)
    );
    // Bun.write does not preserve the executable bit.
    await run(["chmod", "755", resolve(staging, "bin", executableName())]);

    const installation = resolve(staging, "installation.json");
    await writeFile(
      installation,
      `${Schema.encodeSync(Schema.parseJson(IcnInstallationDeclaration))({
        schemaVersion: 1,
        // One backend now, and `cpu` is the honest name for it: vLLM in the EIM image serves
        // from system memory on Xeon cores. The variant is retained rather than renamed because
        // it crosses the generated protocol.
        backend: "cpu",
        nativeBuild: identity.native_build,
        backendModuleAbi: identity.backend_module_abi,
      })}\n`
    );

    await rm(destination, { recursive: true, force: true });
    await rename(staging, destination);
    return {
      installationPath: resolve(destination, "installation.json"),
      binaryPath: resolve(destination, "bin", executableName()),
    };
  } catch (cause) {
    await rm(staging, { recursive: true, force: true });
    throw cause;
  }
};

if (import.meta.main) {
  const result = await buildLocalIcn();
  if (process.argv.includes("--serve")) {
    const child = Bun.spawn(
      [
        result.binaryPath,
        "serve",
        "--installation",
        result.installationPath,
        ...process.argv.slice(2).filter((argument) => argument !== "--serve"),
      ],
      {
        cwd: PROJECT_ROOT,
        env: process.env,
        stdin: "inherit",
        stdout: "inherit",
        stderr: "inherit",
      }
    );
    process.exit(await child.exited);
  }
  console.log(
    `ICN development installation ready at ${result.installationPath}`
  );
}
