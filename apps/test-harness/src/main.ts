import { volten, Buffer, RawBuffer, Kernel, struct, array } from '@volten/core';

// Expose all public API on window for Playwright's page.evaluate()
(window as any).Volten = { volten, Buffer, RawBuffer, Kernel, struct, array };

// Probe GPU readiness so tests can await it
(async () => {
    try {
        // Just test that we can get a device — don't hold onto it
        const v = await volten();
        (window as any).__gpuReady = true;
        (window as any).__voltenInstance = v;
        document.getElementById('status')!.textContent = 'GPU Ready ✓';
    } catch (e) {
        (window as any).__gpuReady = false;
        (window as any).__gpuError = String(e);
        document.getElementById('status')!.textContent = 'GPU Failed: ' + e;
    }
})();
