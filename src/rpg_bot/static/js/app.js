/**
 * Alpine.js chat application component.
 */
function chatApp() {
  return {
    chats: [],
    activeChatId: null,
    messages: [],
    userInput: '',
    streaming: false,
    streamingContent: '',
    gameSystems: [],
    activeGameSystem: null,
    sidebarOpen: true,
    editingTitle: null,
    editTitleValue: '',
    apiKey: null,

    async init() {
      this.apiKey = localStorage.getItem('apiKey');
      await this.loadGameSystems();
      await this.loadChats();

      // Restore last active chat from localStorage
      const lastChatId = localStorage.getItem('activeChatId');
      if (lastChatId && this.chats.find(c => c.id === lastChatId)) {
        await this.selectChat(lastChatId);
      }
    },

    async apiFetch(path, opts = {}, retried = false) {
      const headers = { ...(opts.headers || {}) };
      if (this.apiKey) headers['Authorization'] = 'Bearer ' + this.apiKey;
      const resp = await window.fetch(path, { ...opts, headers });
      if (resp.status === 401 && !retried) {
        const key = prompt('This server requires an API key (API_KEY in .env). Enter it:');
        if (key && key.trim()) {
          localStorage.setItem('apiKey', key.trim());
          this.apiKey = key.trim();
          return this.apiFetch(path, opts, true);
        }
      }
      return resp;
    },

    async loadGameSystems() {
      try {
        const resp = await this.apiFetch('/api/game-systems');
        const data = await resp.json();
        this.gameSystems = data.game_systems || [];
      } catch (e) {
        console.error('Failed to load game systems:', e);
      }
    },

    async loadChats() {
      try {
        const resp = await this.apiFetch('/api/chats');
        this.chats = await resp.json();
      } catch (e) {
        console.error('Failed to load chats:', e);
      }
    },

    async newChat() {
      try {
        const resp = await this.apiFetch('/api/chats', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ game_system: this.activeGameSystem }),
        });
        const chat = await resp.json();
        this.chats.unshift(chat);
        await this.selectChat(chat.id);
      } catch (e) {
        console.error('Failed to create chat:', e);
      }
    },

    async selectChat(id) {
      if (this.streaming) return;
      this.activeChatId = id;
      localStorage.setItem('activeChatId', id);

      try {
        const resp = await this.apiFetch(`/api/chats/${id}`);
        const chat = await resp.json();
        this.messages = chat.messages || [];
        this.activeGameSystem = chat.game_system;
        this.$nextTick(() => this.scrollToBottom());
      } catch (e) {
        console.error('Failed to load chat:', e);
      }
    },

    async deleteChat(id) {
      try {
        await this.apiFetch(`/api/chats/${id}`, { method: 'DELETE' });
        this.chats = this.chats.filter(c => c.id !== id);
        if (this.activeChatId === id) {
          this.activeChatId = null;
          this.messages = [];
        }
      } catch (e) {
        console.error('Failed to delete chat:', e);
      }
    },

    async updateGameSystem(gameSystem) {
      this.activeGameSystem = gameSystem || null;
      if (!this.activeChatId) return;
      try {
        await this.apiFetch(`/api/chats/${this.activeChatId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ game_system: this.activeGameSystem }),
        });
        const chat = this.chats.find(c => c.id === this.activeChatId);
        if (chat) chat.game_system = this.activeGameSystem;
      } catch (e) {
        console.error('Failed to update game system:', e);
      }
    },

    startEditTitle(chatId) {
      const chat = this.chats.find(c => c.id === chatId);
      if (!chat) return;
      this.editingTitle = chatId;
      this.editTitleValue = chat.title;
    },

    async saveTitle(chatId) {
      this.editingTitle = null;
      if (!this.editTitleValue.trim()) return;
      try {
        const resp = await this.apiFetch(`/api/chats/${chatId}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: this.editTitleValue.trim() }),
        });
        const updated = await resp.json();
        const chat = this.chats.find(c => c.id === chatId);
        if (chat) chat.title = updated.title;
      } catch (e) {
        console.error('Failed to update title:', e);
      }
    },

    async sendMessage() {
      const text = this.userInput.trim();
      if (!text || this.streaming) return;

      // Create a chat if none is active
      if (!this.activeChatId) {
        await this.newChat();
      }

      // Add user message to UI immediately
      this.messages.push({ role: 'user', content: text });
      this.userInput = '';
      this.streaming = true;
      this.streamingContent = '';
      this.$nextTick(() => this.scrollToBottom());

      // Build message history for the API
      const apiMessages = this.messages.map(m => ({
        role: m.role,
        content: m.content,
      }));

      try {
        const resp = await this.apiFetch('/v1/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            model: 'rpg-bot',
            messages: apiMessages,
            stream: true,
            game_system: this.activeGameSystem,
            chat_id: this.activeChatId,
          }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop();

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            const data = line.slice(6);
            if (data === '[DONE]') continue;
            try {
              const parsed = JSON.parse(data);
              const content = parsed.choices?.[0]?.delta?.content;
              if (content) {
                this.streamingContent += content;
                this.$nextTick(() => this.scrollToBottom());
              }
            } catch (e) {
              // Skip malformed chunks
            }
          }
        }

        // Move streaming content to messages
        if (this.streamingContent) {
          this.messages.push({ role: 'assistant', content: this.streamingContent });
        }
      } catch (e) {
        console.error('Streaming failed:', e);
        this.messages.push({
          role: 'assistant',
          content: 'Error: Failed to get response. Please try again.',
        });
      } finally {
        this.streaming = false;
        this.streamingContent = '';
        // Refresh chat list to get updated title
        await this.loadChats();
      }
    },

    renderMarkdown(text) {
      if (!text) return '';
      const html = marked.parse(text, { breaks: true });
      return linkifyCitations(html);
    },

    scrollToBottom() {
      const el = document.getElementById('messages');
      if (el) el.scrollTop = el.scrollHeight;
    },

    handleKeydown(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        this.sendMessage();
      }
    },

    get activeChat() {
      return this.chats.find(c => c.id === this.activeChatId);
    },
  };
}
