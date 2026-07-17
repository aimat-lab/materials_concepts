import { dataLoader } from "./data-loader.js";
import { AdjacencyIndex } from "./adjacency.js";

export class StaticPredictor {
  constructor() {
    this.baselineSession = null;
    this.gnnSession = null;
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
    
    // Default load baseline model
    await this.ensureModelLoaded("baseline", onProgress);

    this.isInitialized = true;
    onProgress({ step: "ready", percent: 100, text: "Inference engine ready!" });
  }

  async ensureModelLoaded(modelKey = "baseline", onProgress = () => {}) {
    if (modelKey === "baseline") {
      if (this.baselineSession) return;
      const modelPath = dataLoader.manifest.models.baseline.onnx;
      const modelUrl = `${dataLoader.baseDataUrl}/${modelPath}`;
      const modelBuffer = await (await fetch(modelUrl)).arrayBuffer();
      this.baselineSession = await globalThis.ort.InferenceSession.create(
        new Uint8Array(modelBuffer),
        { executionProviders: ["wasm"] }
      );
    } else if (modelKey === "gnn") {
      // Lazy load GNN embeddings binary files if not already loaded
      await dataLoader.loadGNNData(onProgress);

      if (this.gnnSession) return;
      const modelPath = dataLoader.manifest.models.gnn.onnx;
      const modelUrl = `${dataLoader.baseDataUrl}/${modelPath}`;
      const modelBuffer = await (await fetch(modelUrl)).arrayBuffer();
      this.gnnSession = await globalThis.ort.InferenceSession.create(
        new Uint8Array(modelBuffer),
        { executionProviders: ["wasm"] }
      );
    } else {
      throw new Error(`Unknown model key '${modelKey}'`);
    }
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

  async predict(conceptName, k = 200, maxDegree = null, onStatus = () => {}, modelKey = "baseline") {
    if (!this.isInitialized) {
      throw new Error("Predictor not initialized. Call init() first.");
    }

    await this.ensureModelLoaded(modelKey, (prog) => onStatus(prog.text));

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

    const scores = new Float32Array(totalPairs);

    if (modelKey === "gnn") {
      await this.predictGNN(conceptId, candidates, scores, onStatus);
    } else {
      await this.predictBaseline(conceptId, candidates, scores, onStatus);
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

  async predictBaseline(srcId, candidates, scores, onStatus) {
    const totalPairs = candidates.length;
    const srcOffset = srcId * 10;
    const srcFeat = dataLoader.features.subarray(srcOffset, srcOffset + 10);

    const batchSize = 25000;
    const totalBatches = Math.ceil(totalPairs / batchSize);

    for (let b = 0; b < totalBatches; b++) {
      const start = b * batchSize;
      const end = Math.min(start + batchSize, totalPairs);
      const currentBatchSize = end - start;

      onStatus(`Scoring pairs ${start.toLocaleString()} - ${end.toLocaleString()} of ${totalPairs.toLocaleString()} (Baseline MLP)...`);

      const inputData = new Float32Array(currentBatchSize * 20);

      for (let i = 0; i < currentBatchSize; i++) {
        const targetId = candidates[start + i];
        const tgtOffset = targetId * 10;

        const rowOffset = i * 20;
        inputData.set(srcFeat, rowOffset);
        inputData.subarray(rowOffset + 10, rowOffset + 20).set(
          dataLoader.features.subarray(tgtOffset, tgtOffset + 10)
        );
      }

      const inputTensor = new globalThis.ort.Tensor("float32", inputData, [currentBatchSize, 20]);
      const results = await this.baselineSession.run({ input: inputTensor });
      scores.set(results.output.data, start);

      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }

  async predictGNN(srcId, candidates, scores, onStatus) {
    const totalPairs = candidates.length;
    const embDim = 128;
    const inputDim = 512; // [z_u, z_v, abs(z_u - z_v), z_u * z_v]

    const srcOffset = srcId * embDim;
    const z_u = dataLoader.gnnEmbeddings.subarray(srcOffset, srcOffset + embDim);

    const batchSize = 20000;
    const totalBatches = Math.ceil(totalPairs / batchSize);

    for (let b = 0; b < totalBatches; b++) {
      const start = b * batchSize;
      const end = Math.min(start + batchSize, totalPairs);
      const currentBatchSize = end - start;

      onStatus(`Scoring pairs ${start.toLocaleString()} - ${end.toLocaleString()} of ${totalPairs.toLocaleString()} (GraphSAGE GNN)...`);

      const inputData = new Float32Array(currentBatchSize * inputDim);

      for (let i = 0; i < currentBatchSize; i++) {
        const targetId = candidates[start + i];
        const tgtOffset = targetId * embDim;
        const z_v = dataLoader.gnnEmbeddings.subarray(tgtOffset, tgtOffset + embDim);

        const rowOffset = i * inputDim;

        // 1. z_u (0..127)
        inputData.set(z_u, rowOffset);

        // 2. z_v (128..255)
        inputData.subarray(rowOffset + 128, rowOffset + 256).set(z_v);

        // 3. abs(z_u - z_v) (256..383)
        // 4. z_u * z_v (384..511)
        for (let j = 0; j < embDim; j++) {
          const zuVal = z_u[j];
          const zvVal = z_v[j];
          inputData[rowOffset + 256 + j] = Math.abs(zuVal - zvVal);
          inputData[rowOffset + 384 + j] = zuVal * zvVal;
        }
      }

      const inputTensor = new globalThis.ort.Tensor("float32", inputData, [currentBatchSize, inputDim]);
      const results = await this.gnnSession.run({ input: inputTensor });
      scores.set(results.output.data, start);

      await new Promise(resolve => setTimeout(resolve, 0));
    }
  }
}

export const predictor = new StaticPredictor();
