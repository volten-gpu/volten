import { execFileSync } from 'node:child_process';
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readdirSync,
    rmSync,
    writeFileSync
} from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const rootDir = resolve(scriptDir, '..');
const tempDir = mkdtempSync(join(tmpdir(), 'volten-pack-smoke-'));
const packDir = join(tempDir, 'pack');
const consumerDir = join(tempDir, 'consumer');

const npm = process.platform === 'win32' ? 'npm.cmd' : 'npm';
const node = process.execPath;
const tsc =
    process.platform === 'win32'
        ? join(rootDir, 'node_modules', '.bin', 'tsc.cmd')
        : join(rootDir, 'node_modules', '.bin', 'tsc');

function run(command, args, options = {}) {
    execFileSync(command, args, {
        cwd: rootDir,
        stdio: 'inherit',
        shell: process.platform === 'win32' && command.endsWith('.cmd'),
        ...options
    });
}

try {
    mkdirSync(packDir, { recursive: true });
    mkdirSync(consumerDir, { recursive: true });

    run(npm, [
        'pack',
        '--silent',
        '--pack-destination',
        packDir,
        join(rootDir, 'packages', 'core')
    ]);

    const tarball = readdirSync(packDir)
        .filter((name) => name.endsWith('.tgz'))
        .map((name) => join(packDir, name))[0];

    if (!tarball) {
        throw new Error('npm pack did not produce a tarball.');
    }

    writeFileSync(
        join(consumerDir, 'package.json'),
        JSON.stringify(
            {
                private: true,
                type: 'module'
            },
            null,
            2
        )
    );

    run(npm, ['install', '--no-audit', '--no-fund', tarball], {
        cwd: consumerDir
    });

    writeFileSync(
        join(consumerDir, 'runtime-smoke.mjs'),
        `
const mod = await import('@volten/core');
const expected = ['volten', 'Buffer', 'RawBuffer', 'Uniform', 'kernel', 'plan', 'struct', 'array', 'unpack'];
const missing = expected.filter((name) => !(name in mod));

if (missing.length > 0) {
    throw new Error(\`Missing public exports: \${missing.join(', ')}\`);
}
`
    );

    run(node, [join(consumerDir, 'runtime-smoke.mjs')], {
        cwd: consumerDir
    });

    writeFileSync(
        join(consumerDir, 'tsconfig.json'),
        JSON.stringify(
            {
                compilerOptions: {
                    target: 'ES2022',
                    module: 'ESNext',
                    moduleResolution: 'Bundler',
                    strict: true,
                    skipLibCheck: false,
                    lib: ['ES2022', 'DOM']
                },
                include: ['index.ts']
            },
            null,
            2
        )
    );

    writeFileSync(
        join(consumerDir, 'index.ts'),
        `
import {
    Buffer,
    kernel,
    plan,
    RawBuffer,
    Uniform,
    type InvocationOptions,
    type KernelConfig,
    type OperationContext,
    type ReadTarget,
    type VoltenOptions
} from '@volten/core';

const options: VoltenOptions = {};
const maybeDevice: GPUDevice | undefined = options.device;
const target: ReadTarget | undefined = undefined;
const invocationOptions: InvocationOptions = { label: 'smoke node' };
const buffer = new Buffer([1, 2, 3], 'f32');
const raw = new RawBuffer(new Uint32Array([1]).buffer, 'array<u32>');
const uniform = new Uniform(1, 'f32');
const config: KernelConfig = {
    shader: 'fn main(gid: vec3u) {}',
    threads: 1
};
const operation = kernel(config);
const node = operation({ buffer }, invocationOptions);
const composed = plan((context: OperationContext, inputs: { buffer: Buffer }) => {
    void context.device;
    const step = operation({ buffer: inputs.buffer });
    return { result: step.buffer };
});
const planNode = composed({ buffer });

void [maybeDevice, target, buffer, raw, uniform, node, planNode];
`
    );

    if (!existsSync(tsc)) {
        throw new Error(`TypeScript binary not found at ${tsc}`);
    }

    run(tsc, ['--noEmit', '--project', join(consumerDir, 'tsconfig.json')], {
        cwd: consumerDir
    });

    console.log('Pack smoke test passed.');
} finally {
    rmSync(tempDir, { recursive: true, force: true });
}
