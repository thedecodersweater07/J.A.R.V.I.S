import { configureStore } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import appReducer from '../features/app/appSlice';
import chatReducer from '../features/chat/chatSlice';
import settingsReducer from '../features/settings/settingsSlice';

export const store = configureStore({
  reducer: {
    app: appReducer,
    chat: chatReducer,
    settings: settingsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['chat/messageReceived'],
        // Ignore these field paths in all actions
        ignoredActionPaths: ['payload.timestamp'],
        // Ignore these paths in the state
        ignoredPaths: ['chat.messages'],
      },
    }),
});

// Infer the `RootState` and `AppDispatch` types from the store itself
export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

// Use throughout your app instead of plain `useDispatch` and `useSelector`
export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
