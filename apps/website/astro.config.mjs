import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
    site: 'https://volten-gpu.github.io',
    integrations: [
        starlight({
            title: 'Volten',
            description: 'WebGPU compute, without the ceremony.',
            logo: {
                src: './src/assets/logo.svg',
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
                }
            ]
        })
    ]
});
