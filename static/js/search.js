import { dataLoader } from "./data-loader.js";

export class StaticSearch {
  async search(query, k = 200, ignoreCase = true) {
    if (!dataLoader.isLoaded) {
      throw new Error("DataLoader not initialized");
    }

    const trimmed = query.trim();
    if (!trimmed) return [];

    const q = ignoreCase ? trimmed.toLowerCase() : trimmed;
    const lookup = dataLoader.lookup;
    const matches = [];

    for (let i = 0; i < lookup.length; i++) {
      const item = lookup[i];
      const concept = item.concept;
      const conceptStr = ignoreCase ? concept.toLowerCase() : concept;

      if (conceptStr.includes(q)) {
        matches.push(item);
      }
    }

    // Sort by count descending
    matches.sort((a, b) => b.count - a.count);

    if (k && k > 0) {
      return matches.slice(0, k);
    }

    return matches;
  }

  async getConcepts(query = "", limit = 200, page = 1) {
    if (!dataLoader.isLoaded) {
      throw new Error("DataLoader not initialized");
    }

    let items = dataLoader.lookup;
    const q = query.trim().toLowerCase();

    if (q) {
      items = items.filter(item => item.concept.toLowerCase().includes(q));
    }

    const total = items.length;
    const start = (page - 1) * limit;
    const paged = items.slice(start, start + limit);

    return {
      total,
      page,
      limit,
      items: paged.map(item => item.concept)
    };
  }
}

export const staticSearch = new StaticSearch();
