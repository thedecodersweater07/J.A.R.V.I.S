import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../../store/store';

export interface AppState {
  isInitialized: boolean;
  isLoading: boolean;
  error: string | null;
  version: string;
  isConnected: boolean;
  lastUpdated: number | null;
}

const initialState: AppState = {
  isInitialized: false,
  isLoading: false,
  error: null,
  version: '1.0.0',
  isConnected: false,
  lastUpdated: null,
};

// Async thunks
export const initializeApp = createAsyncThunk(
  'app/initialize',
  async (_, { dispatch }) => {
    try {
      // Initialize any required services here
      // For example, check Python service status
      const response = await fetch('/api/status');
      if (!response.ok) {
        throw new Error('Failed to connect to backend services');
      }
      
      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Initialization error:', error);
      throw error;
    }
  }
);

const appSlice = createSlice({
  name: 'app',
  initialState,
  reducers: {
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.isLoading = action.payload;
    },
    setError: (state, action: PayloadAction<string | null>) => {
      state.error = action.payload;
    },
    setConnected: (state, action: PayloadAction<boolean>) => {
      state.isConnected = action.payload;
    },
    resetApp: () => initialState,
  },
  extraReducers: (builder) => {
    builder
      .addCase(initializeApp.pending, (state) => {
        state.isLoading = true;
        state.error = null;
      })
      .addCase(initializeApp.fulfilled, (state, action) => {
        state.isInitialized = true;
        state.isLoading = false;
        state.isConnected = true;
        state.lastUpdated = Date.now();
        state.version = action.payload?.version || state.version;
      })
      .addCase(initializeApp.rejected, (state, action) => {
        state.isLoading = false;
        state.error = action.error.message || 'Initialization failed';
        state.isConnected = false;
      });
  },
});

// Selectors
export const selectIsInitialized = (state: RootState) => state.app.isInitialized;
export const selectIsLoading = (state: RootState) => state.app.isLoading;
export const selectError = (state: RootState) => state.app.error;
export const selectIsConnected = (state: RootState) => state.app.isConnected;

// Actions
export const { setLoading, setError, setConnected, resetApp } = appSlice.actions;

export default appSlice.reducer;
