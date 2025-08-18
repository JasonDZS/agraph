interface EnvConfig {
  APP_TITLE: string;
  API_BASE_URL: string;
  APP_ENV: 'development' | 'production' | 'test';
  ENABLE_DEBUG: boolean;
  LOG_LEVEL: 'debug' | 'info' | 'warn' | 'error';
}

class Environment {
  private config: EnvConfig;

  constructor() {
    this.config = {
      APP_TITLE: import.meta.env.VITE_APP_TITLE || 'AGraph',
      API_BASE_URL:
        import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
      APP_ENV: import.meta.env.VITE_APP_ENV || 'development',
      ENABLE_DEBUG: import.meta.env.VITE_ENABLE_DEBUG === 'true',
      LOG_LEVEL: import.meta.env.VITE_LOG_LEVEL || 'info',
    };

    // Validate required environment variables
    this.validateConfig();
  }

  private validateConfig() {
    const requiredVars = ['APP_TITLE', 'API_BASE_URL'];

    for (const key of requiredVars) {
      if (!this.config[key as keyof EnvConfig]) {
        throw new Error(`Missing required environment variable: VITE_${key}`);
      }
    }
  }

  get<K extends keyof EnvConfig>(key: K): EnvConfig[K] {
    return this.config[key];
  }

  isDevelopment(): boolean {
    return this.config.APP_ENV === 'development';
  }

  isProduction(): boolean {
    return this.config.APP_ENV === 'production';
  }

  isTest(): boolean {
    return this.config.APP_ENV === 'test';
  }

  getConfig(): Readonly<EnvConfig> {
    return Object.freeze({ ...this.config });
  }
}

export const env = new Environment();
export type { EnvConfig };
