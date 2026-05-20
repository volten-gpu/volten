import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
    site: 'https://volten-gpu.github.io',
    integrations: [
        starlight({
            title: 'Volten',
            description: 'WebGPU compute, without the ceremony.',
            favicon: '/favicon.png',
            logo: {
                src: './src/assets/logo.png',
                alt: 'Volten'
            },
            social: [
                {
                    icon: 'github',
                    label: 'GitHub',
                    href: 'https://github.com/volten-gpu/volten'
                }
            ],
            editLink: {
                baseUrl:
                    'https://github.com/volten-gpu/volten/edit/main/apps/website/'
            },
            sidebar: [
                {
                    label: 'Start Here',
                    items: [
                        { slug: 'index', label: 'Overview' },
                        { slug: 'getting-started' },
                        { slug: 'guides/core-concepts' }
                    ]
                },
                {
                    label: 'Data',
                    items: [
                        { slug: 'data/buffers' },
                        { slug: 'data/uniforms' },
                        { slug: 'data/raw-buffers' },
                        { slug: 'data/structs-and-arrays' },
                        { slug: 'data/reading-data' }
                    ]
                },
                {
                    label: 'Kernels',
                    items: [
                        { slug: 'kernels/writing-kernels' },
                        { slug: 'kernels/bindings' },
                        { slug: 'kernels/outputs' },
                        { slug: 'kernels/threads-and-workgroups' },
                        { slug: 'kernels/bounds-guards' }
                    ]
                },
                {
                    label: 'Execution',
                    items: [
                        { slug: 'execution/overview', label: 'Overview' },
                        { slug: 'execution/pass' },
                        { slug: 'execution/run' },
                        { slug: 'execution/wait' },
                        { slug: 'execution/nodes-and-handles' },
                        { slug: 'execution/read' },
                        { slug: 'execution/destroy' }
                    ]
                },
                {
                    label: 'Debugging',
                    items: [
                        { slug: 'debugging/overview', label: 'Overview' },
                        { slug: 'debugging/shader-logs' },
                        { slug: 'debugging/read-debug' }
                    ]
                },
                {
                    label: 'Examples',
                    items: [
                        { slug: 'examples', label: 'Overview' },
                        { slug: 'examples/scale-values' },
                        { slug: 'examples/map-input-to-output' },
                        { slug: 'examples/uniform-parameters' },
                        { slug: 'examples/run-two-nodes' },
                        { slug: 'examples/debug-one-invocation' }
                    ]
                }
            ]
        })
    ]
});
