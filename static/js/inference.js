import { dataLoader } from "./data-loader.js";
import { AdjacencyIndex } from "./adjacency.js";

export class StaticPredictor {
  constructor() {
    this.session = null;
    this.adjacency = null;
    this.isInitialized = false;
  }

  async init(onProgress = () => {}) {
    if (this.isInitialized) return;

    await dataLoader.loadAll(onProgress);
    this.adjacency = new AdjacencyIndex(
      dataLoader.indptr,
      dataLoader.indices,
      dataLoader.degrees,
      dataLoader.vertices
    );

    onProgress({ step: "onnx", percent: 95, text: "Initializing ONNX Runtime engine..." });

    // Ensure ort (ONNX Runtime Web) is loaded
    if (typeof globalThis.ort === "undefined") {
      throw new Error("ONNX Runtime Web library (ort.min.js) not found. Please check HTML script tags.");
    }

    // Configure ONNX Runtime
    globalThis.ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    
    // Load ONNX model with external weight data if present
    const modelUrl = `${dataLoader.baseDataUrl}/${dataLoader.manifest.model}`;
    
    try {
      // Fetch model proto bytes & external data bytes to ensure seamless browser loading
      const modelBuffer = await (await fetch(modelUrl)).arrayBuffer();
      const externalDataUrl = `${modelUrl}.data`;
      
      let sessionOptions = { executionProviders: ["wasm"] };
      try {
        const extRes = await fetch(externalDataUrl);
        if (extRes.ok) {
          const extBuffer = await extRes.arrayBuffer();
          sessionOptions.externalData = [
            {
              path: "baseline.onnx.data",
              data: new Uint8Array(extBuffer)
            }
          ];
        }
      } catch (e) {
        console.warn("No external weight file or failed to fetch, attempting direct session load", e);
      }

      this.session = await globalThis.ort.InferenceSession.create(
        new Uint8Array(modelBuffer),
        sessionOptions
      );
    } catch (err) {
      console.error("Failed to initialize ONNX session:", err);
      throw err;
    }

    this.isInitialized = true;
    onProgress({ step: "ready", percent: 100, text: "Inference engine ready!" });
  }

  getPairs(conceptId, maxDegree = null) {
    const vertices = this.adjacency.vertices;
    const neighborSet = this.adjacency.neighborSet(conceptId);
    const unconnected = [];

    for (let i = 0; i < vertices.length; i++) {
      const other = vertices[i];
      if (other === conceptId) continue;
      if (maxDegree !== null && this.adjacency.degree(other) > maxDegree) continue;
      if (neighborSet.has(other)) continue;

      unconnected.push(other);
    }

    return unconnected;
  }

  async predict(conceptName, k = 200, maxDegree = null, onStatus = () => {}) {
    if (!this.isInitialized) {
      throw new Error("Predictor not initialized. Call init() first.");
    }

    const conceptId = dataLoader.conceptToId[conceptName];
    if (conceptId === undefined) {
      throw new Error(`Unknown concept '${conceptName}'`);
    }

    onStatus("Filtering candidate concept pairs...");
    const candidates = this.getPairs(conceptId, maxDegree);
    const totalPairs = candidates.length;

    if (totalPairs === 0) {
      return [];
    }

    onStatus(`Scoring ${totalPairs.toLocaleString()} candidate pairs...`);

    // Extract source concept feature vector (10 float32 elements)
    const srcOffset = conceptId * 10;
    const srcFeat = dataLoader.features.subarray(srcOffset, srcOffset + 10);

    const scores = new Float32Array(totalPairs);
    const batchSize = 25000; // Process 25k pairs per ONNX forward call
    const totalBatches = Math.ceil(totalPairs / batchSize);

    for (let b = 0; b < totalBatches; b++) {
      const start = b * batchSize;
      const end = Math.min(start + batchSize, totalPairs);
      const currentBatchSize = end - start;

      onStatus(`Scoring pairs ${start.toLocaleString()} - ${end.toLocaleString()} of ${totalPairs.toLocaleString()}...`);

      // Construct input tensor: (currentBatchSize, 20)
      const inputData = new Float32Array(currentBatchSize * 20);

      for (let i = 0; i < currentBatchSize; i++) {
        const targetId = candidates[start + i];
        const tgtOffset = targetId * 10;

        const rowOffset = i * 20;
        // Copy 10 features of source concept
        inputData.set(srcFeat, rowOffset);
        // Copy 10 features of target concept
        inputData.subarray(rowOffset + 10, rowOffset + 20).set(
          dataLoader.features.subarray(tgtOffset, tgtOffset + 10)
        );
      }

      const inputTensor = new globalThis.ort.Tensor("float32", inputData, [currentBatchSize, 20]);
      const results = await this.session.run({ input: inputTensor });
      const outputData = results.output.data;

      scores.set(outputData, start);

      // Give UI thread breathing room
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    onStatus("Sorting predictions by relevance...");

    // Create array of index pairs [idx, score] to sort
    const items = new Array(totalPairs);
    for (let i = 0; i < totalPairs; i++) {
      items[i] = { candidateId: candidates[i], score: scores[i] };
    }

    items.sort((a, b) => b.score - a.score);

    const topK = items.slice(0, k).map(item => ({
      concept: dataLoader.idToConcept.get(item.candidateId),
      score: item.score
    }));

    return topK;
  }
}

export const predictor = new StaticPredictor();
