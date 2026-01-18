import { name as coreName } from '@volten/core';
import { name as stdlibName } from '@volten/stdlib';

console.log(`Volten Playground Initialized.`);
console.log(`Loaded Core: ${coreName}`);
console.log(`Loaded Stdlib: ${stdlibName}`);

async function initWebGPU() {
  if (!navigator.gpu) {
    console.error("WebGPU not supported in this browser.");
    return;
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    console.error("No WebGPU adapter found.");
    return;
  }
  const device = await adapter.requestDevice();
  console.log("WebGPU Device created:", device);
}

initWebGPU();
