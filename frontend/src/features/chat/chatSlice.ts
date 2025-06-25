import { createAsyncThunk, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { RootState } from '../../store/store';

export interface Message {
  id: string;
  sender: 'user' | 'jarvis';
  content: string;
  timestamp: number;
  status: 'sending' | 'sent' | 'error';
  type?: 'text' | 'code' | 'image' | 'file';
  language?: string;
}

export interface ChatState {
  messages: Message[];
  isTyping: boolean;
  input: string;
  activeConversation: string | null;
  conversations: {
    id: string;
    title: string;
    lastMessage: number;
  }[];
  error: string | null;
}

const initialState: ChatState = {
  messages: [],
  isTyping: false,
  input: '',
  activeConversation: null,
  conversations: [],
  error: null,
};

// Helper function to generate a unique ID
const generateId = () => Math.random().toString(36).substring(2, 11);

// Async thunks
export const sendMessage = createAsyncThunk<
  Message,
  { content: string; conversationId?: string },
  { state: RootState }
>(
  'chat/sendMessage',
  async ({ content, conversationId }, { getState }) => {
    try {
      const response = await fetch('/api/chat/message', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          content,
          conversationId: conversationId || getState().chat.activeConversation,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send message');
      }

      return await response.json();
    } catch (error) {
      console.error('Error sending message:', error);
      throw error;
    }
  }
);

export const loadConversation = createAsyncThunk<
  { messages: Message[]; conversationId: string },
  string,
  { state: RootState }
>('chat/loadConversation', async (conversationId) => {
  try {
    const response = await fetch(`/api/chat/conversation/${conversationId}`);
    if (!response.ok) {
      throw new Error('Failed to load conversation');
    }
    return await response.json();
  } catch (error) {
    console.error('Error loading conversation:', error);
    throw error;
  }
});

const chatSlice = createSlice({
  name: 'chat',
  initialState,
  reducers: {
    setInput: (state, action: PayloadAction<string>) => {
      state.input = action.payload;
    },
    clearInput: (state) => {
      state.input = '';
    },
    startTyping: (state) => {
      state.isTyping = true;
    },
    stopTyping: (state) => {
      state.isTyping = false;
    },
    addMessage: (state, action: PayloadAction<Omit<Message, 'id' | 'timestamp' | 'status'>>) => {
      const newMessage: Message = {
        ...action.payload,
        id: generateId(),
        timestamp: Date.now(),
        status: 'sent',
      };
      state.messages.push(newMessage);
    },
    setActiveConversation: (state, action: PayloadAction<string | null>) => {
      state.activeConversation = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(sendMessage.pending, (state, action) => {
        const tempId = `temp-${Date.now()}`;
        state.messages.push({
          id: tempId,
          sender: 'user',
          content: action.meta.arg.content,
          timestamp: Date.now(),
          status: 'sending',
        });
      })
      .addCase(sendMessage.fulfilled, (state, action) => {
        const index = state.messages.findIndex((msg) => msg.status === 'sending');
        if (index !== -1) {
          state.messages[index] = {
            ...action.payload,
            status: 'sent',
          };
        } else {
          state.messages.push({
            ...action.payload,
            status: 'sent',
          });
        }
      })
      .addCase(sendMessage.rejected, (state, action) => {
        const index = state.messages.findIndex((msg) => msg.status === 'sending');
        if (index !== -1) {
          state.messages[index].status = 'error';
        }
        state.error = action.error.message || 'Failed to send message';
      })
      .addCase(loadConversation.fulfilled, (state, action) => {
        state.messages = action.payload.messages;
        state.activeConversation = action.payload.conversationId;
      })
      .addCase(loadConversation.rejected, (state, action) => {
        state.error = action.error.message || 'Failed to load conversation';
      });
  },
});

// Selectors
export const selectMessages = (state: RootState) => state.chat.messages;
export const selectIsTyping = (state: RootState) => state.chat.isTyping;
export const selectInput = (state: RootState) => state.chat.input;
export const selectActiveConversation = (state: RootState) =>
  state.chat.activeConversation;
export const selectConversations = (state: RootState) =>
  state.chat.conversations;
export const selectChatError = (state: RootState) => state.chat.error;

// Actions
export const {
  setInput,
  clearInput,
  startTyping,
  stopTyping,
  addMessage,
  setActiveConversation,
  clearError,
} = chatSlice.actions;

export default chatSlice.reducer;