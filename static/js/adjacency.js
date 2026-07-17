/**
 * AdjacencyIndex Class
 * High-performance CSR graph index operations using TypedArrays
 */
export class AdjacencyIndex {
  constructor(indptr, indices, degrees, vertices) {
    this.indptr = indptr;
    this.indices = indices;
    this.degrees = degrees;
    this.vertices = vertices;
  }

  degree(nodeId) {
    return this.degrees[nodeId] || 0;
  }

  neighbors(nodeId) {
    const start = this.indptr[nodeId];
    const end = this.indptr[nodeId + 1];
    return this.indices.subarray(start, end);
  }

  neighborSet(nodeId) {
    const arr = this.neighbors(nodeId);
    return new Set(arr);
  }

  hasEdge(src, dst) {
    const row = this.neighbors(src);
    if (row.length === 0) return false;
    
    // Binary search because neighbor indices are sorted in CSR
    let low = 0;
    let high = row.length - 1;
    while (low <= high) {
      const mid = (low + high) >>> 1;
      const val = row[mid];
      if (val === dst) return true;
      if (val < dst) low = mid + 1;
      else high = mid - 1;
    }
    return false;
  }
}
