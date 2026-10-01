export interface KeywordEvent {
  keyword: string;
  confidence: number;
  timestampSec: number;
}

export interface HealthStatus {
  score: number;
  severity: number;
}

export interface DetectionEvent {
  keyword: string;
  confidence: number;
}

export declare class SononShm {
  constructor(path?: string, sampleRate?: number, capacity?: number);
  writeSamples(samples: ArrayLike<number>): number;
  readHealth(): HealthStatus;
  readDetection(): DetectionEvent | null;
  close(): void;
}

export declare class SononEngine {
  constructor(sampleRate?: number, frameSize?: number, hopSize?: number, numMfcc?: number);
  extractFeatures(samples: ArrayLike<number>): number[][];
  enrollKeyword(name: string, features: number[][], threshold?: number): void;
  ingestSamples(samples: ArrayLike<number>): KeywordEvent[];
}

export declare const SHM_MAGIC: number;
export declare const SHM_VERSION: number;
export declare const DEFAULT_SHM_PATH: string;
