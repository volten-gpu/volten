import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './e2e',
    timeout: 30_000,

    // Retry once on failure (GPU tests can occasionally flake)
    retries: 1,

    use: {
        channel: 'chrome',
        launchOptions: {
            args: [
                '--enable-unsafe-webgpu',
            ],
        },
        // No need for screenshots/videos for compute tests,
        // but enable on failure for debugging
        screenshot: 'only-on-failure',
        video: 'off',
    },

    // Auto-start the test-harness dev server before tests run
    webServer: {
        command: 'cd apps/test-harness && npx vite --port 5174',
        port: 5174,
        reuseExistingServer: !process.env.CI,
        timeout: 15_000,
    },
});
