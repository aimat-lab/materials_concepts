/**
 * Data Loader Module
 * Handles loading binary arrays, JSON tables, ONNX model and caching them in memory.
 */

class DataLoader {
  constructor(baseDataUrl = "./data") {
    this.baseDataUrl = baseDataUrl;
    this.manifest = null;
    this.features = null;
    this.indptr = null;
    this.degrees = null;
    this.vertices = null;
    this.indices = null;
    this.lookup = null;
    this.conceptToId = null;
    this.idToConcept = null;
    this.gnnEmbeddings = null;
    this.isLoaded = false;
    this.isGNNLoaded = false;
  }

  async loadAll(onProgress = () => {}) {
    if (this.isLoaded) {
      return this;
    }

    // 1. Fetch Manifest
    onProgress({ step: "manifest", percent: 5, text: "Fetching data manifest..." });
    const manifestRes = await fetch(`${this.baseDataUrl}/manifest.json`);
    this.manifest = await manifestRes.json();

    // 2. Fetch Lookup & Concept Map
    onProgress({ step: "lookup", percent: 15, text: "Loading concept lookup tables..." });
    const [lookupRes, c2idRes] = await Promise.all([
      fetch(`${this.baseDataUrl}/${this.manifest.lookup}`),
      fetch(`${this.baseDataUrl}/${this.manifest.concept_to_id}`)
    ]);

    this.lookup = await lookupRes.json();
    this.conceptToId = await c2idRes.json();
    
    // Create reverse ID->Concept lookup array/map
    this.idToConcept = new Map();
    for (const item of this.lookup) {
      this.idToConcept.set(item.id, item.concept);
    }

    // 3. Fetch Features
    onProgress({ step: "features", percent: 30, text: "Loading feature vectors (5.2 MB)..." });
    const featBuf = await (await fetch(`${this.baseDataUrl}/${this.manifest.features}`)).arrayBuffer();
    this.features = new Float32Array(featBuf);

    // 4. Fetch Adjacency Header files (indptr, degrees, vertices)
    onProgress({ step: "adjacency_meta", percent: 45, text: "Loading graph metadata..." });
    const adjMeta = this.manifest.adjacency;
    const [indptrBuf, degreesBuf, verticesBuf] = await Promise.all([
      fetch(`${this.baseDataUrl}/${adjMeta.indptr}`).then(r => r.arrayBuffer()),
      fetch(`${this.baseDataUrl}/${adjMeta.degrees}`).then(r => r.arrayBuffer()),
      fetch(`${this.baseDataUrl}/${adjMeta.vertices}`).then(r => r.arrayBuffer())
    ]);

    this.indptr = new Int32Array(indptrBuf);
    this.degrees = new Int32Array(degreesBuf);
    this.vertices = new Int32Array(verticesBuf);

    // 5. Fetch Adjacency Indices Chunks and combine
    const chunks = adjMeta.indices_chunks;
    this.indices = new Int32Array(adjMeta.total_indices);
    let offset = 0;

    for (let i = 0; i < chunks.length; i++) {
      const chunkPercent = 50 + Math.round(((i + 1) / chunks.length) * 45);
      onProgress({
        step: `adjacency_chunk_${i}`,
        percent: chunkPercent,
        text: `Loading graph edges part ${i + 1}/${chunks.length}...`
      });

      const chunkBuf = await (await fetch(`${this.baseDataUrl}/${chunks[i]}`)).arrayBuffer();
      const chunkArr = new Int32Array(chunkBuf);
      this.indices.set(chunkArr, offset);
      offset += chunkArr.length;
    }

    onProgress({ step: "complete", percent: 100, text: "Data loading complete!" });
    this.isLoaded = true;
    return this;
  }

  async loadGNNData(onProgress = () => {}) {
    if (this.isGNNLoaded) {
      return this;
    }

    if (!this.isLoaded) {
      await this.loadAll(onProgress);
    }

    const gnnMeta = this.manifest.models?.gnn;
    if (!gnnMeta) {
      throw new Error("GraphSAGE GNN metadata missing from manifest.json");
    }

    const totalElements = gnnMeta.total_nodes * gnnMeta.emb_dim;
    this.gnnEmbeddings = new Float32Array(totalElements);

    const chunks = gnnMeta.embedding_chunks;
    let offset = 0;

    for (let i = 0; i < chunks.length; i++) {
      const percent = Math.round(((i + 1) / chunks.length) * 100);
      onProgress({
        step: `gnn_chunk_${i}`,
        percent: percent,
        text: `Loading GraphSAGE embeddings part ${i + 1}/${chunks.length} (~22.3 MB raw each)...`
      });

      const chunkBuf = await (await fetch(`${this.baseDataUrl}/${chunks[i]}`)).arrayBuffer();
      const chunkArr = new Float32Array(chunkBuf);
      this.gnnEmbeddings.set(chunkArr, offset);
      offset += chunkArr.length;
    }

    this.isGNNLoaded = true;
    return this;
  }
}

// Global singleton instance
export const dataLoader = new DataLoader();
