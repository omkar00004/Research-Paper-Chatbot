/**
 * Scholar -  RAG Research Assistant
 * Client-side logic: conversations, chat, uploads, theme toggle
 */

// ── State ─────────────────────────────────────────────
let currentConversationId = null;
let conversations = [];
let papers = [];
let busy = false;

// ── DOM References ────────────────────────────────────
const chatArea = document.getElementById('chatArea');
const chatMessages = document.getElementById('chatMessages');
const emptyState = document.getElementById('emptyState');
const composerInput = document.getElementById('composerInput');
const sendBtn = document.getElementById('sendBtn');
const conversationList = document.getElementById('conversationList');
const emptyConversations = document.getElementById('emptyConversations');
const paperList = document.getElementById('paperList');
const noPapers = document.getElementById('noPapers');
const headerTitle = document.getElementById('headerTitle');
const libraryLabel = document.getElementById('libraryLabel');
const statChunks = document.getElementById('statChunks');
const statPapers = document.getElementById('statPapers');
const paperCountFooter = document.getElementById('paperCountFooter');
const dropZone = document.getElementById('dropZone');

// ── Theme Management ──────────────────────────────────

function getPreferredTheme() {
  const stored = localStorage.getItem('scholar-theme');
  if (stored) return stored;
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('scholar-theme', theme);

  const sunIcon = document.getElementById('sunIcon');
  const moonIcon = document.getElementById('moonIcon');
  if (theme === 'dark') {
    sunIcon.style.display = '';
    moonIcon.style.display = 'none';
  } else {
    sunIcon.style.display = 'none';
    moonIcon.style.display = '';
  }
}

function toggleTheme() {
  const current = document.documentElement.getAttribute('data-theme') || 'dark';
  applyTheme(current === 'dark' ? 'light' : 'dark');
}

// ── Mobile Sidebar ────────────────────────────────────

function toggleSidebar() {
  const sidebar = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebarOverlay');
  sidebar.classList.toggle('open');
  overlay.classList.toggle('open');
}

// ── API Helpers ───────────────────────────────────────

const API = {
  async get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`GET ${url} failed: ${res.status}`);
    return res.json();
  },
  async post(url, body) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`POST ${url} failed: ${res.status}`);
    return res.json();
  },
  async del(url) {
    const res = await fetch(url, { method: 'DELETE' });
    if (!res.ok) throw new Error(`DELETE ${url} failed: ${res.status}`);
    return res.json();
  },
  async upload(url, file) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(url, { method: 'POST', body: formData });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },
};

// ── Conversations ─────────────────────────────────────

async function loadConversations() {
  try {
    conversations = await API.get('/api/conversations');
    renderConversationList();
  } catch (e) {
    console.warn('Failed to load conversations:', e);
  }
}

function renderConversationList() {
  if (!conversations.length) {
    conversationList.innerHTML = '<div class="empty-conversations">No conversations yet.</div>';
    return;
  }

  conversationList.innerHTML = conversations.map(c => `
    <div class="conversation-item ${c.id === currentConversationId ? 'active' : ''}"
         onclick="switchConversation('${c.id}')" data-id="${c.id}">
      <span class="conversation-item-icon">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
      </span>
      <div class="conversation-item-content">
        <div class="conversation-item-title">${escapeHtml(c.title)}</div>
        <div class="conversation-item-time mono">${formatTime(c.updated_at)} · ${c.message_count} msgs</div>
      </div>
      <button class="conversation-item-delete" onclick="event.stopPropagation(); deleteConversation('${c.id}')" title="Delete">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
  `).join('');
}

async function newConversation() {
  try {
    const conv = await API.post('/api/conversations', { title: 'New conversation' });
    currentConversationId = conv.id;
    await loadConversations();
    clearChat();
    headerTitle.textContent = 'New conversation';
    composerInput.focus();
  } catch (e) {
    console.error('Failed to create conversation:', e);
  }
}

async function switchConversation(convId) {
  if (convId === currentConversationId) return;
  currentConversationId = convId;

  try {
    const conv = await API.get(`/api/conversations/${convId}`);
    headerTitle.textContent = conv.title === 'New conversation' ? 'New conversation' : 'Research session';
    renderConversationList();

    if (conv.messages && conv.messages.length > 0) {
      showMessages();
      chatMessages.innerHTML = '';
      for (const msg of conv.messages) {
        if (msg.role === 'user') {
          appendUserBubble(msg.content);
        } else {
          appendAssistantBubble(msg.content, msg.sources, msg.pipeline, msg.time);
        }
      }
      scrollToBottom();
    } else {
      clearChat();
    }
  } catch (e) {
    console.error('Failed to load conversation:', e);
  }

  // Close mobile sidebar
  document.getElementById('sidebar').classList.remove('open');
  document.getElementById('sidebarOverlay').classList.remove('open');
}

async function deleteConversation(convId) {
  try {
    await API.del(`/api/conversations/${convId}`);
    if (convId === currentConversationId) {
      currentConversationId = null;
      clearChat();
      headerTitle.textContent = 'New conversation';
    }
    await loadConversations();
  } catch (e) {
    console.error('Failed to delete conversation:', e);
  }
}

// ── Chat Rendering ────────────────────────────────────

function clearChat() {
  chatMessages.innerHTML = '';
  chatMessages.style.display = 'none';
  emptyState.style.display = 'flex';
}

function showMessages() {
  emptyState.style.display = 'none';
  chatMessages.style.display = 'flex';
}

function appendUserBubble(text) {
  showMessages();
  const div = document.createElement('div');
  div.className = 'user-bubble';
  div.innerHTML = `<div class="user-bubble-inner">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
}

function appendAssistantBubble(text, sources, pipeline, time) {
  showMessages();
  const div = document.createElement('div');
  div.className = 'assistant-bubble';

  let html = `
    <div class="assistant-avatar">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.2 3.6a1 1 0 0 0 .6.6L17.4 8.4 13.8 9.6a1 1 0 0 0-.6.6L12 13.8 10.8 10.2a1 1 0 0 0-.6-.6L6.6 8.4 10.2 7.2a1 1 0 0 0 .6-.6L12 3z"/><path d="M18 14l.8 2.4a.5.5 0 0 0 .3.3L21.5 17.5l-2.4.8a.5.5 0 0 0-.3.3L18 21l-.8-2.4a.5.5 0 0 0-.3-.3L14.5 17.5l2.4-.8a.5.5 0 0 0 .3-.3L18 14z"/></svg>
    </div>
    <div class="assistant-content">
      <div class="assistant-text">${formatMarkdown(text)}</div>
  `;

  // Sources/citations
  if (sources && sources.length > 0) {
    html += `
      <div class="citations-section">
        <div class="citations-label mono">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2H4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/><path d="M15 21c3 0 7-1 7-8V5c0-1.25-.756-2.017-2-2h-4c-1.25 0-2 .75-2 1.972V11c0 1.25.75 2 2 2 1 0 1 0 1 1v1c0 1-1 2-2 2s-1 .008-1 1.031V21z"/></svg>
          Sources · ${sources.length}
        </div>
        <div class="citations-list">
          ${sources.map((s, i) => `
            <div class="citation-card">
              <div class="citation-card-header">
                <div class="citation-card-left">
                  <span class="citation-badge mono">${i + 1}</span>
                  <span class="citation-file-icon">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
                  </span>
                  <span class="citation-paper-name">${escapeHtml(s.paper)}</span>
                </div>
                <span class="citation-page mono">p.${s.page}</span>
              </div>
              ${s.section ? `<div class="citation-snippet serif">${escapeHtml(s.section)}</div>` : ''}
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  // Pipeline timings
  if (pipeline && pipeline.length > 0) {
    html += `
      <div class="timings-section">
        ${pipeline.map(p => `
          <div class="timing-badge mono">
            <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
            ${escapeHtml(p.step)}
            <span class="timing-value">${(p.time * 1000).toFixed(0)}ms</span>
          </div>
        `).join('')}
        ${time ? `
          <div class="timing-badge mono">
            <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>
            total
            <span class="timing-value">${(time * 1000).toFixed(0)}ms</span>
          </div>
        ` : ''}
      </div>
    `;
  }

  html += '</div>';
  div.innerHTML = html;
  chatMessages.appendChild(div);
}

function appendStreamingPlaceholder() {
  showMessages();
  const div = document.createElement('div');
  div.className = 'assistant-bubble';
  div.id = 'streamingBubble';
  div.innerHTML = `
    <div class="assistant-avatar">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.2 3.6a1 1 0 0 0 .6.6L17.4 8.4 13.8 9.6a1 1 0 0 0-.6.6L12 13.8 10.8 10.2a1 1 0 0 0-.6-.6L6.6 8.4 10.2 7.2a1 1 0 0 0 .6-.6L12 3z"/><path d="M18 14l.8 2.4a.5.5 0 0 0 .3.3L21.5 17.5l-2.4.8a.5.5 0 0 0-.3.3L18 21l-.8-2.4a.5.5 0 0 0-.3-.3L14.5 17.5l2.4-.8a.5.5 0 0 0 .3-.3L18 14z"/></svg>
    </div>
    <div class="assistant-content">
      <div class="streaming-indicator">
        <span class="streaming-label mono">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
          Retrieving context
        </span>
        <span class="pulse-dots">
          <span class="pulse-dot"></span>
          <span class="pulse-dot"></span>
          <span class="pulse-dot"></span>
        </span>
      </div>
    </div>
  `;
  chatMessages.appendChild(div);
}

function removeStreamingPlaceholder() {
  const el = document.getElementById('streamingBubble');
  if (el) el.remove();
}

// ── Send Message ──────────────────────────────────────

async function sendMessage() {
  const text = composerInput.value.trim();
  if (!text || busy) return;

  busy = true;
  sendBtn.disabled = true;
  composerInput.value = '';
  composerInput.style.height = 'auto';

  // Auto-create conversation if none active
  if (!currentConversationId) {
    try {
      const conv = await API.post('/api/conversations', { title: 'New conversation' });
      currentConversationId = conv.id;
    } catch (e) {
      console.error('Failed to create conversation:', e);
      busy = false;
      updateSendButton();
      return;
    }
  }

  // Show user bubble
  appendUserBubble(text);
  scrollToBottom();

  // Show streaming placeholder
  appendStreamingPlaceholder();
  scrollToBottom();

  try {
    const result = await API.post('/api/query', {
      question: text,
      conversation_id: currentConversationId,
    });

    removeStreamingPlaceholder();
    appendAssistantBubble(
      result.answer || 'No response generated.',
      result.sources || [],
      result.pipeline || [],
      result.time || 0,
    );
    scrollToBottom();

    // Refresh conversation list (title may have changed)
    await loadConversations();
  } catch (e) {
    removeStreamingPlaceholder();
    appendAssistantBubble(
      'An error occurred while processing your question. Please try again.',
      [], [], 0,
    );
    console.error('Query error:', e);
  }

  busy = false;
  updateSendButton();
  composerInput.focus();
}

function sendSuggestion(text) {
  composerInput.value = text;
  sendMessage();
}

// ── File Upload ───────────────────────────────────────

async function handleFileUpload(files) {
  if (!files || !files.length) return;

  for (const file of files) {
    if (!file.name.toLowerCase().endsWith('.pdf')) continue;

    showToast(`Uploading ${file.name}...`);
    try {
      const result = await API.upload('/api/upload', file);
      if (result.status === 'success') {
        showToast(`✅ ${file.name}: ${result.chunks} chunks indexed`);
      } else {
        showToast(`⚠️ ${result.message || 'Upload failed'}`);
      }
    } catch (e) {
      showToast(`❌ Failed to upload ${file.name}`);
      console.error('Upload error:', e);
    }
  }

  await loadPapers();
}

// ── Papers / Stats ────────────────────────────────────

async function loadPapers() {
  try {
    const stats = await API.get('/api/stats');
    papers = stats.papers || [];
    const totalChunks = stats.total_chunks || 0;
    const totalPapers = stats.total_papers || 0;

    statChunks.textContent = totalChunks.toLocaleString();
    statPapers.textContent = totalPapers.toLocaleString();
    libraryLabel.textContent = `Library · ${totalPapers}`;
    paperCountFooter.textContent = `answers cited from ${totalPapers} papers`;

    renderPaperList();
  } catch (e) {
    console.warn('Failed to load papers:', e);
  }
}

function renderPaperList() {
  if (!papers.length) {
    paperList.innerHTML = '<div class="no-papers">No papers indexed yet.</div>';
    return;
  }

  paperList.innerHTML = papers.map(name => `
    <div class="paper-item">
      <span class="paper-item-icon">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
      </span>
      <div class="paper-item-info">
        <div class="paper-item-name">${escapeHtml(name)}</div>
      </div>
      <button class="paper-item-delete" onclick="deletePaper('${escapeHtml(name)}')" title="Remove">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>
  `).join('');
}

async function deletePaper(name) {
  try {
    await API.del(`/api/papers/${encodeURIComponent(name)}`);
    showToast(`Deleted ${name}`);
    await loadPapers();
  } catch (e) {
    console.error('Failed to delete paper:', e);
  }
}

// ── UI Helpers ────────────────────────────────────────

function handleKeyDown(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function autoResize(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = Math.min(textarea.scrollHeight, 160) + 'px';
  updateSendButton();
}

function updateSendButton() {
  sendBtn.disabled = busy || !composerInput.value.trim();
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTo({ top: chatArea.scrollHeight, behavior: 'smooth' });
  });
}

function showToast(message) {
  // Remove existing toast
  const existing = document.querySelector('.upload-toast');
  if (existing) existing.remove();

  const toast = document.createElement('div');
  toast.className = 'upload-toast';
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => toast.remove(), 3000);
}

function escapeHtml(str) {
  if (!str) return '';
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

function formatMarkdown(text) {
  if (!text) return '';
  if (typeof marked !== 'undefined') {
    return marked.parse(text);
  }
  let html = escapeHtml(text);
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
  html = html.replace(/`(.*?)`/g, '<code style="background:var(--code-bg);padding:2px 6px;border-radius:4px;font-size:0.9em;">$1</code>');
  html = html.replace(/\n/g, '<br>');
  return html;
}

function formatTime(isoString) {
  if (!isoString) return '';
  try {
    const date = new Date(isoString);
    const now = new Date();
    const diff = now - date;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'just now';
    if (minutes < 60) return `${minutes}m ago`;
    if (hours < 24) return `${hours}h ago`;
    if (days < 7) return `${days}d ago`;
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
}

// ── Drag & Drop ───────────────────────────────────────

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('dragging');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('dragging');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragging');
  handleFileUpload(e.dataTransfer.files);
});

// ── Input listener ────────────────────────────────────
composerInput.addEventListener('input', () => updateSendButton());

// ── Initialize ────────────────────────────────────────

async function init() {
  applyTheme(getPreferredTheme());
  await Promise.all([
    loadConversations(),
    loadPapers(),
  ]);
  composerInput.focus();
}

init();
