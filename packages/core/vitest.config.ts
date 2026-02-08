import { defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        // Run tests in Node.js environment (not browser)
        environment: 'node',

        // Test file patterns
        include: ['tests/**/*.test.ts'],

        // Enable globals like describe, it, expect without imports
        globals: true,

        // Coverage configuration (optional - run with --coverage)
        coverage: {
            provider: 'v8',
            include: ['src/**/*.ts'],
            exclude: ['src/**/index.ts'],
        },
    },
});
