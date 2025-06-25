import { useEffect } from 'react';
import { useAppDispatch } from '../store/hooks';
import { sendMessage } from '../features/chat/chatSlice';

type ShortcutHandler = (e: KeyboardEvent) => void;

const useKeyboardShortcuts = () => {
  const dispatch = useAppDispatch();

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't trigger shortcuts when typing in inputs or textareas
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Focus search input on Cmd+K / Ctrl+K
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        const searchInput = document.querySelector('input[type="search"]') as HTMLInputElement;
        if (searchInput) {
          searchInput.focus();
        }
      }

      // New chat on Cmd+N / Ctrl+N
      if ((e.metaKey || e.ctrlKey) && e.key === 'n') {
        e.preventDefault();
        // Handle new chat
      }

      // Focus chat input on / key
      if (e.key === '/' && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        const chatInput = document.querySelector('textarea') as HTMLTextAreaElement;
        if (chatInput) {
          chatInput.focus();
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [dispatch]);
};

export default useKeyboardShortcuts;
