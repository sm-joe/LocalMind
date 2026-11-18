"use client";

import {
  ChangeEvent,
  FormEvent,
  useEffect,
  useRef,
  useState,
} from "react";

type Source = {
  document_id: string;
  filename: string;
  chunk_id: number;
  score: number;
  citation: string;
};

type ChatResponse = {
  answer: string;
  sources: Source[];
  conversation_id: string | null;
  user_message_id: string | null;
  assistant_message_id: string | null;
};

type Document = {
  document_id: string;
  filename: string;
  content_type: string;
  chunks: number;
};

type DocumentListResponse = {
  documents: Document[];
};

type UploadResponse = {
  document_id: string;
  filename: string;
  chunks: number;
  status: string;
};

type ConversationSummary = {
  id: string;
  title: string;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
  message_count: number;
};

type ConversationDetail = {
  id: string;
  title: string;
  is_archived: boolean;
  created_at: string;
  updated_at: string;
};

type StoredMessage = {
  id: string;
  sequence_no: number;
  role: "user" | "assistant";
  content: string;
  status: "pending" | "complete" | "failed";
  created_at: string;
  sources: Source[];
};

type MessagePage = {
  items: StoredMessage[];
  next_cursor: number | null;
};

type ConversationListResponse = {
  items: ConversationSummary[];
  next_cursor: string | null;
};

type Message = {
  id: string;
  sequence_no?: number;
  role: "user" | "assistant";
  content: string;
  status?: "pending" | "complete" | "failed";
  sources?: Source[];
};

const API_URL = "/api/backend";

const ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt", ".md"];

async function responseError(
  response: Response,
  fallback: string,
): Promise<Error> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") {
      return new Error(body.detail);
    }
    if (body.detail && typeof body.detail === "object") {
      return new Error(JSON.stringify(body.detail));
    }
  } catch {
    // Keep the operation-specific fallback for non-JSON responses.
  }
  return new Error(fallback);
}

function Workspace({ onLogout }: { onLogout: () => void }) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [conversationSearch, setConversationSearch] = useState("");
  const [documentSearch, setDocumentSearch] = useState("");
  const [uploadNotice, setUploadNotice] = useState<string | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [copyNotice, setCopyNotice] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [conversationsLoading, setConversationsLoading] = useState(true);
  const [conversationNextCursor, setConversationNextCursor] = useState<string | null>(null);
  const [loadingMoreConversations, setLoadingMoreConversations] = useState(false);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [conversationError, setConversationError] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  const [evidencePanelCollapsed, setEvidencePanelCollapsed] = useState(false);
  const [conversationActionId, setConversationActionId] = useState<string | null>(null);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [documentError, setDocumentError] = useState<string | null>(null);
  const [sourcePreview, setSourcePreview] = useState<{
    document_id: string; filename: string; chunk_id: number; text: string;
  } | null>(null);
  const [sourcePreviewLoading, setSourcePreviewLoading] = useState(false);
  const [sourcePreviewError, setSourcePreviewError] = useState<string | null>(null);


  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const initialLoadStartedRef = useRef(false);
  const messageLoadRequestRef = useRef(0);

  async function previewSource(source: Source) {
    setSourcePreview(null);
    setSourcePreviewError(null);
    setSourcePreviewLoading(true);
    setEvidencePanelCollapsed(false);
    try {
      const response = await fetch(
        `${API_URL}/documents/${encodeURIComponent(source.document_id)}/chunks/${source.chunk_id}`,
        { cache: "no-store" },
      );
      if (!response.ok) throw await responseError(response, "Could not load source passage.");
      const data = (await response.json()) as {
        document_id: string; filename: string; chunk_id: number; text: string;
      };
      setSourcePreview(data);
    } catch (error) {
      setSourcePreviewError(error instanceof Error ? error.message : "Could not load source passage.");
    } finally {
      setSourcePreviewLoading(false);
    }
  }

  async function loadDocuments() {
    setDocumentsLoading(true);
    setDocumentError(null);
    try {
      const response = await fetch(`${API_URL}/documents`);
      if (!response.ok) {
        throw await responseError(response, "Failed to load documents.");
      }
      const data = (await response.json()) as DocumentListResponse;
      setDocuments(data.documents);
    } catch (requestError) {
      setDocumentError(
        requestError instanceof Error ? requestError.message : "Failed to load documents.",
      );
    } finally {
      setDocumentsLoading(false);
    }
  }

  async function fetchConversationList(cursor?: string | null): Promise<ConversationListResponse> {
    const query = new URLSearchParams({ limit: "100", include_archived: "true" });
    if (cursor) query.set("cursor", cursor);
    const response = await fetch(`${API_URL}/conversations?${query.toString()}`);
    if (!response.ok) {
      throw await responseError(response, "Failed to load conversation history.");
    }
    return (await response.json()) as ConversationListResponse;
  }

  async function loadConversations() {
    setConversationsLoading(true);
    setConversationError(null);
    try {
      const page = await fetchConversationList();
      setConversations(page.items);
      setConversationNextCursor(page.next_cursor);
      return page.items;
    } catch (requestError) {
      setConversationError(
        requestError instanceof Error
          ? requestError.message
          : "Failed to load conversation history.",
      );
      return null;
    } finally {
      setConversationsLoading(false);
    }
  }

  async function loadMoreConversations() {
    if (!conversationNextCursor || loadingMoreConversations) return;
    setLoadingMoreConversations(true);
    setConversationError(null);
    try {
      const page = await fetchConversationList(conversationNextCursor);
      setConversations((current) => {
        const knownIds = new Set(current.map((item) => item.id));
        return [...current, ...page.items.filter((item) => !knownIds.has(item.id))];
      });
      setConversationNextCursor(page.next_cursor);
    } catch (requestError) {
      setConversationError(
        requestError instanceof Error
          ? requestError.message
          : "Failed to load more conversations.",
      );
    } finally {
      setLoadingMoreConversations(false);
    }
  }

  async function fetchAllMessages(conversationId: string): Promise<Message[]> {
    const allMessages: StoredMessage[] = [];
    let beforeSequence: number | null = null;
    const seenCursors = new Set<number>();

    do {
      const query = new URLSearchParams({ limit: "100" });
      if (beforeSequence !== null) {
        query.set("before_sequence", String(beforeSequence));
      }
      const response = await fetch(
        `${API_URL}/conversations/${encodeURIComponent(conversationId)}/messages?${query.toString()}`,
      );
      if (!response.ok) {
        throw await responseError(response, "Failed to restore conversation messages.");
      }
      const page = (await response.json()) as MessagePage;
      allMessages.unshift(...page.items);
      beforeSequence = page.next_cursor;
      if (beforeSequence !== null) {
        if (seenCursors.has(beforeSequence)) {
          throw new Error("Conversation history pagination returned a repeated cursor.");
        }
        seenCursors.add(beforeSequence);
      }
    } while (beforeSequence !== null);

    return allMessages.map((message) => ({
      id: message.id,
      sequence_no: message.sequence_no,
      role: message.role,
      content: message.content,
      status: message.status,
      sources: message.sources ?? [],
    }));
  }

  async function openConversation(conversationId: string) {
    const requestId = ++messageLoadRequestRef.current;
    setActiveConversationId(conversationId);
    setMessages([]);
    setError(null);
    setConversationError(null);
    setMessagesLoading(true);
    try {
      const restoredMessages = await fetchAllMessages(conversationId);
      if (requestId === messageLoadRequestRef.current) {
        setMessages(restoredMessages);
      }
    } catch (requestError) {
      if (requestId === messageLoadRequestRef.current) {
        setConversationError(
          requestError instanceof Error
            ? requestError.message
            : "Failed to restore conversation messages.",
        );
      }
    } finally {
      if (requestId === messageLoadRequestRef.current) {
        setMessagesLoading(false);
      }
    }
  }

  useEffect(() => {
    if (initialLoadStartedRef.current) return;
    initialLoadStartedRef.current = true;

    // Initial network requests intentionally populate component state.
    void loadDocuments();
    void (async () => {
      const items = await loadConversations();
      const mostRecentActive = items?.find((item) => !item.is_archived);
      if (mostRecentActive && !activeConversationId) {
        await openConversation(mostRecentActive.id);
      }
    })();
    // Initial load should run once; handlers use the current state when invoked.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isLoading, messagesLoading]);

  async function createConversation() {
    if (isLoading || messagesLoading || conversationActionId) return;
    setConversationError(null);
    setError(null);
    setConversationActionId("new");
    try {
      const response = await fetch(`${API_URL}/conversations`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: "New conversation" }),
      });
      if (!response.ok) {
        throw await responseError(response, "Could not create a new conversation.");
      }
      const conversation = (await response.json()) as ConversationDetail;
      setConversations((current) => [
        {
          ...conversation,
          message_count: 0,
        },
        ...current.filter((item) => item.id !== conversation.id),
      ]);
      messageLoadRequestRef.current += 1;
      setActiveConversationId(conversation.id);
      setMessages([]);
      setQuestion("");
      setMessagesLoading(false);
    } catch (requestError) {
      setConversationError(
        requestError instanceof Error ? requestError.message : "Could not create a new conversation.",
      );
    } finally {
      setConversationActionId(null);
    }
  }

  async function renameConversation(conversation: ConversationSummary) {
    const proposedTitle = window.prompt("Rename conversation", conversation.title);
    if (proposedTitle === null) return;
    const title = proposedTitle.trim();
    if (!title) {
      setConversationError("Conversation title cannot be blank.");
      return;
    }
    if (title.length > 200) {
      setConversationError("Conversation title must be 200 characters or fewer.");
      return;
    }

    setConversationActionId(conversation.id);
    setConversationError(null);
    try {
      const response = await fetch(`${API_URL}/conversations/${encodeURIComponent(conversation.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      });
      if (!response.ok) {
        throw await responseError(response, "Could not rename the conversation.");
      }
      const updated = (await response.json()) as ConversationDetail;
      setConversations((current) => current.map((item) =>
        item.id === updated.id ? { ...item, title: updated.title, updated_at: updated.updated_at } : item,
      ));
    } catch (requestError) {
      setConversationError(requestError instanceof Error ? requestError.message : "Could not rename the conversation.");
    } finally {
      setConversationActionId(null);
    }
  }

  async function toggleArchive(conversation: ConversationSummary) {
    const nextArchived = !conversation.is_archived;
    const action = nextArchived ? "archive" : "restore";
    if (!window.confirm(`${nextArchived ? "Archive" : "Restore"} “${conversation.title}”?`)) return;

    setConversationActionId(conversation.id);
    setConversationError(null);
    try {
      const response = await fetch(`${API_URL}/conversations/${encodeURIComponent(conversation.id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ is_archived: nextArchived }),
      });
      if (!response.ok) {
        throw await responseError(response, `Could not ${action} the conversation.`);
      }
      if (activeConversationId === conversation.id && nextArchived) {
        messageLoadRequestRef.current += 1;
        setActiveConversationId(null);
        setMessages([]);
      }
      if (!nextArchived && showArchived) setShowArchived(false);
      await loadConversations();
    } catch (requestError) {
      setConversationError(
        requestError instanceof Error ? requestError.message : `Could not ${action} the conversation.`,
      );
    } finally {
      setConversationActionId(null);
    }
  }

  async function deleteConversation(conversation: ConversationSummary) {
    if (!window.confirm(`Permanently delete “${conversation.title}” and its messages? This cannot be undone.`)) return;

    setConversationActionId(conversation.id);
    setConversationError(null);
    try {
      const response = await fetch(`${API_URL}/conversations/${encodeURIComponent(conversation.id)}`, {
        method: "DELETE",
      });
      if (!response.ok) {
        throw await responseError(response, "Could not delete the conversation.");
      }
      if (activeConversationId === conversation.id) {
        messageLoadRequestRef.current += 1;
        setActiveConversationId(null);
        setMessages([]);
      }
      setConversations((current) => current.filter((item) => item.id !== conversation.id));
    } catch (requestError) {
      setConversationError(requestError instanceof Error ? requestError.message : "Could not delete the conversation.");
    } finally {
      setConversationActionId(null);
    }
  }

  async function sendMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedQuestion = question.trim();
    if (!trimmedQuestion || isLoading || messagesLoading) return;

    setError(null);
    setConversationError(null);
    setIsLoading(true);

    const optimisticId = `optimistic-${crypto.randomUUID()}`;
    setMessages((current) => [
      ...current,
      { id: optimisticId, role: "user", content: trimmedQuestion, status: "complete" },
    ]);
    setQuestion("");

    let conversationId = activeConversationId;
    try {
      if (!conversationId) {
        const createResponse = await fetch(`${API_URL}/conversations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ title: "New conversation" }),
        });
        if (!createResponse.ok) {
          throw await responseError(createResponse, "Could not create a conversation for this message.");
        }
        const created = (await createResponse.json()) as ConversationDetail;
        conversationId = created.id;
        setActiveConversationId(created.id);
        setConversations((current) => [
          { ...created, message_count: 0 },
          ...current.filter((item) => item.id !== created.id),
        ]);
      }

      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": crypto.randomUUID(),
        },
        body: JSON.stringify({ question: trimmedQuestion, top_k: 5, conversation_id: conversationId }),
      });
      if (!response.ok) {
        throw await responseError(response, "Unable to get a response from LocalRAG.");
      }
      const data = (await response.json()) as ChatResponse;
      setMessages((current) => [
        ...current.filter((message) => message.id !== optimisticId),
        {
          id: data.user_message_id ?? optimisticId,
          role: "user",
          content: trimmedQuestion,
          status: "complete",
        },
        {
          id: data.assistant_message_id ?? `assistant-${crypto.randomUUID()}`,
          role: "assistant",
          content: data.answer,
          sources: data.sources,
          status: "complete",
        },
      ]);
      await loadConversations();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Something went wrong.");
      if (conversationId) {
        try {
          const restored = await fetchAllMessages(conversationId);
          setMessages(restored);
        } catch {
          // Preserve the visible optimistic question and error if history cannot be reloaded.
        }
      } else {
        setMessages((current) => current.filter((message) => message.id !== optimisticId));
      }
    } finally {
      setIsLoading(false);
    }
  }

  function openFilePicker() {
    if (!uploading) fileInputRef.current?.click();
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (file) await uploadDocument(file);
  }

  async function uploadDocument(file: File) {
    const extension = `.${file.name.split(".").pop()?.toLowerCase()}`;
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setDocumentError("Unsupported file type. Use PDF, DOCX, TXT, or MD.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      setDocumentError("File exceeds the 10 MB limit.");
      return;
    }
    setUploading(true);
    setUploadNotice(null);
    setDocumentError(null);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_URL}/documents/upload`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw await responseError(response, "Document upload failed.");
      const data = (await response.json()) as UploadResponse;
      await loadDocuments();
      setDocumentError(null);
      setUploadNotice(data.status === "already_indexed"
        ? `${data.filename} is already in your library.`
        : `${data.filename} added · ${data.chunks} chunks indexed.`);
    } catch (requestError) {
      setDocumentError(requestError instanceof Error ? requestError.message : "Document upload failed.");
    } finally {
      setUploading(false);
    }
  }

  async function deleteDocument(documentId: string) {
    if (deletingDocumentId) return;
    if (!window.confirm("Delete this document and all of its indexed chunks?")) return;
    setDeletingDocumentId(documentId);
    setDocumentError(null);
    try {
      const response = await fetch(`${API_URL}/documents/${encodeURIComponent(documentId)}`, {
        method: "DELETE",
      });
      if (!response.ok) throw await responseError(response, "Failed to delete document.");
      await loadDocuments();
    } catch (requestError) {
      setDocumentError(requestError instanceof Error ? requestError.message : "Failed to delete document.");
    } finally {
      setDeletingDocumentId(null);
    }
  }

  const activeConversation = conversations.find((item) => item.id === activeConversationId);
  const visibleConversations = conversations.filter((item) =>
    item.is_archived === showArchived &&
    item.title.toLowerCase().includes(conversationSearch.trim().toLowerCase()),
  );
  const visibleDocuments = documents.filter((item) =>
    item.filename.toLowerCase().includes(documentSearch.trim().toLowerCase()),
  );
  const evidenceSources = [...messages]
    .reverse()
    .find((message) => message.role === "assistant" && (message.sources?.length ?? 0) > 0)
    ?.sources ?? [];
  const conversationControlsDisabled = isLoading || messagesLoading || conversationActionId !== null;

  async function copyText(content: string) {
    try {
      await navigator.clipboard.writeText(content);
      setCopyNotice("Copied to clipboard");
    } catch {
      setCopyNotice("Could not access clipboard.");
    }
  }

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="app-header product-header">
          <div className="brand-lockup"><span className="brand-mark" aria-hidden="true">◆</span><span>localrag <span className="brand-mark-subtle">/ workspace</span></span></div>
          <div className="header-actions"><span className="status-pill"><span className="status-dot" />Private · Local</span><button type="button" onClick={onLogout} className="header-signout">Sign out ↗</button></div>
        </header>
        <div className="workspace-intro">
          <div><p className="eyebrow">YOUR KNOWLEDGE, CONNECTED</p><h1>Good questions start here<span className="heading-dot">.</span></h1><p className="subtitle">Explore documents, find trustworthy answers, and trace every insight to its source.</p></div>
          <div className="workspace-metrics" aria-label="Workspace overview"><div><strong>{documents.length}</strong><span>Documents</span></div><div><strong>{conversations.filter((item) => !item.is_archived).length}</strong><span>Conversations</span></div></div>
        </div>

        <div className="workspace-grid">
          <aside className="sidebar-panel">
            <section className="conversations-section">
              <div className="conversation-panel-header">
                <div>
                  <h2>Conversations</h2>
                  <p>{visibleConversations.length} {visibleConversations.length === 1 ? "chat" : "chats"}</p>
                </div>
                <button
                  type="button"
                  className="new-conversation-button"
                  onClick={() => void createConversation()}
                  disabled={conversationControlsDisabled}
                >
                  + New chat
                </button>
              </div>

              <label className="library-search-wrap"><span aria-hidden="true">⌕</span><input className="library-search" type="search" placeholder="Find a conversation" aria-label="Find a conversation" value={conversationSearch} onChange={(event) => setConversationSearch(event.target.value)} /></label>
              <div className="conversation-toolbar">
                <span className="conversation-section-label">{showArchived ? "Archived chats" : "Recent chats"}</span>
                <button
                  type="button"
                  className="text-button"
                  onClick={() => {
                    setShowArchived((current) => !current);
                  }}
                  disabled={conversationsLoading || conversationControlsDisabled}
                >
                  {showArchived ? "Show active" : "Archived"}
                </button>
              </div>

              {conversationError && <div className="conversation-error" role="alert">{conversationError}</div>}

              <div className="conversation-list" aria-label="Conversation history">
                {conversationsLoading ? (
                  <div className="conversation-empty">Loading conversations…</div>
                ) : visibleConversations.length === 0 ? (
                  <div className="conversation-empty">
                    <strong>{conversationSearch ? "No matching conversations" : showArchived ? "No archived chats" : "No conversations yet"}</strong>
                    <span>{conversationSearch ? "Try a different search term." : showArchived ? "Archived chats will appear here." : "Start a new chat to keep your questions and answers."}</span>
                  </div>
                ) : visibleConversations.map((conversation) => (
                  <article
                    key={conversation.id}
                    className={`conversation-item${activeConversationId === conversation.id ? " conversation-item-active" : ""}`}
                  >
                    <button
                      type="button"
                      className="conversation-select"
                      onClick={() => void openConversation(conversation.id)}
                      disabled={conversationControlsDisabled}
                      aria-current={activeConversationId === conversation.id ? "page" : undefined}
                    >
                      <span className="conversation-title">{conversation.title}</span>
                      <span className="conversation-meta">
                        {conversation.message_count} {conversation.message_count === 1 ? "message" : "messages"}
                        <span aria-hidden="true"> · </span>
                        {new Date(conversation.updated_at).toLocaleDateString()}
                      </span>
                    </button>
                    <div className="conversation-actions">
                      {!conversation.is_archived && (
                        <button
                          type="button"
                          className="conversation-action"
                          title="Rename conversation"
                          aria-label={`Rename ${conversation.title}`}
                          disabled={conversationControlsDisabled || conversationActionId === conversation.id}
                          onClick={() => void renameConversation(conversation)}
                        >Edit</button>
                      )}
                      <button
                        type="button"
                        className="conversation-action"
                        title={conversation.is_archived ? "Restore conversation" : "Archive conversation"}
                        aria-label={`${conversation.is_archived ? "Restore" : "Archive"} ${conversation.title}`}
                        disabled={conversationControlsDisabled || conversationActionId === conversation.id}
                        onClick={() => void toggleArchive(conversation)}
                      >{conversation.is_archived ? "Restore" : "Archive"}</button>
                      <button
                        type="button"
                        className="conversation-action conversation-action-danger"
                        title="Delete conversation"
                        aria-label={`Delete ${conversation.title}`}
                        disabled={conversationControlsDisabled || conversationActionId === conversation.id}
                        onClick={() => void deleteConversation(conversation)}
                      >Delete</button>
                    </div>
                  </article>
                ))}
                {!conversationsLoading && conversationNextCursor && (
                  <button
                    type="button"
                    className="load-more-conversations"
                    onClick={() => void loadMoreConversations()}
                    disabled={loadingMoreConversations || conversationControlsDisabled}
                  >
                    {loadingMoreConversations ? "Loading…" : "Load older conversations"}
                  </button>
                )}
              </div>
            </section>

            <aside className="documents-panel">
              <div className="panel-header">
                <div>
                  <h2>Documents</h2>
                  <p>{documents.length} {documents.length === 1 ? "document" : "documents"}</p>
                </div>
                <button type="button" className="upload-button" onClick={openFilePicker} disabled={uploading}>
                  {uploading ? "Indexing…" : "+ Upload"}
                </button>
                <input ref={fileInputRef} type="file" className="hidden-file-input" accept=".pdf,.docx,.txt,.md" onChange={handleFileChange} />
              </div>
              {documentError && <div className="document-error" role="alert">{documentError}</div>}
              {uploadNotice && <div className="upload-notice" role="status">{uploadNotice}<button type="button" onClick={() => setUploadNotice(null)} aria-label="Dismiss upload notice">×</button></div>}
              {documents.length > 0 && <label className="library-search-wrap"><span aria-hidden="true">⌕</span><input className="library-search" type="search" aria-label="Find a document" placeholder="Find a document" value={documentSearch} onChange={(event) => setDocumentSearch(event.target.value)} /></label>}
              <div className={`document-dropzone${dragActive ? " is-dragging" : ""}`} onDragOver={(event) => { event.preventDefault(); if (!uploading) setDragActive(true); }} onDragLeave={(event) => { event.preventDefault(); setDragActive(false); }} onDrop={(event) => { event.preventDefault(); setDragActive(false); if (!uploading && event.dataTransfer.files[0]) void uploadDocument(event.dataTransfer.files[0]); }}>
              <div className="document-list">
                {documentsLoading ? (
                  <div className="documents-empty">Loading documents…</div>
                ) : documents.length === 0 ? (
                  <div className="documents-empty">
                    <div className="documents-empty-icon">+</div>
                    <strong>No documents yet</strong>
                    <span>Upload a document to start asking questions.</span>
                  </div>
                ) : visibleDocuments.length === 0 ? (<div className="documents-empty"><strong>No matching documents</strong><span>Try another filename.</span></div>) : visibleDocuments.map((document) => (
                  <article className="document-card" key={document.document_id}>
                    <div className="document-icon">DOC</div>
                    <div className="document-info">
                      <span className="document-name">{document.filename}</span>
                      <span className="document-meta">{document.chunks} {document.chunks === 1 ? "chunk" : "chunks"}</span>
                    </div>
                    <button
                      type="button"
                      className="delete-button"
                      aria-label={`Delete ${document.filename}`}
                      title={`Delete ${document.filename}`}
                      disabled={deletingDocumentId === document.document_id}
                      onClick={() => void deleteDocument(document.document_id)}
                    >{deletingDocumentId === document.document_id ? "…" : "×"}</button>
                  </article>
                ))}
              </div>
              <div className="upload-hint">{dragActive ? "Release to upload" : uploading ? "Indexing — keep this page open" : "Drop files here or choose Upload · PDF, DOCX, TXT, MD · 10 MB max"}</div></div>
            </aside>
          </aside>

          <section className="chat-panel">
            <div className="chat-panel-heading">
              <div>
                <span className="chat-context-label">CURRENT CONVERSATION</span>
                <h2>{activeConversation?.title ?? "New conversation"}</h2>
              </div>
              {activeConversation && <span className="chat-message-count">{messages.length} {messages.length === 1 ? "message" : "messages"}</span>}
            </div>

            {messagesLoading ? (
              <div className="conversation-loading-state" role="status">Restoring conversation history…</div>
            ) : messages.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon">✦</div>
                <h2>Turn documents into answers.</h2>
                <p>{documents.length === 0 ? "Upload a document to begin. Your answers will link back to the source passages." : "Ask a question, compare ideas, or explore the evidence in your document library."}</p>
                <div className="suggestions">
                  <button type="button" onClick={() => setQuestion("Summarize the key ideas in my documents.")}>✦ Summarize my documents</button>
                  <button type="button" onClick={() => setQuestion("What are the most important findings?")}>⌕ Find key findings</button>
                </div>
              </div>
            ) : (
              <div className="messages">
                {messages.map((message) => (
                  <article
                    key={message.id}
                    className={`message ${message.role === "user" ? "message-user" : "message-assistant"}`}
                  >
                    <div className="message-label">{message.role === "user" ? "You" : "LocalRAG"}</div>
                    <div className="message-content">
                      {message.status === "failed" && !message.content
                        ? "This response failed to generate. You can ask another question to continue this conversation."
                        : message.content}
                    </div>
                    {message.role === "assistant" && message.status === "complete" && <div className="answer-actions"><button type="button" onClick={() => void copyText(message.content)}>⧉ Copy answer</button></div>}
                    {message.role === "assistant" && message.status === "pending" && (
                      <div className="message-status" role="status">Response pending…</div>
                    )}
                    {message.role === "assistant" && message.status === "failed" && (
                      <div className="message-status message-status-error" role="note">Generation failed</div>
                    )}
                    {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                      <div className="sources">
                        <div className="sources-title">Sources</div>
                        <div className="source-list">
                          {message.sources.map((source, index) => (
                            <div className="source-card" key={`${source.document_id}-${source.chunk_id}-${index}`}>
                              <div className="source-icon">DOC</div>
                              <div className="source-info">
                                <span className="source-name">{source.filename}</span>
                                <span className="source-meta">{source.citation} · Chunk {source.chunk_id} · Score {source.score.toFixed(2)}</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    {message.role === "assistant" && message.status !== "failed" && (!message.sources || message.sources.length === 0) && (
                      <div className="source-status" role="note">
                        <span className="source-status-icon" aria-hidden="true">i</span>
                        <div>
                          <strong>No citations returned</strong>
                          <p>This response has no linked document evidence. Check the uploaded documents before relying on it.</p>
                        </div>
                      </div>
                    )}
                  </article>
                ))}
                {isLoading && (
                  <article className="message message-assistant">
                    <div className="message-label">LocalRAG</div>
                    <div className="typing-indicator" role="status" aria-label="LocalRAG is generating a response"><span /><span /><span /></div>
                  </article>
                )}
                <div ref={messagesEndRef} aria-hidden="true" />
              </div>
            )}

            {error && <div className="error-banner" role="alert"><strong>Request failed.</strong><span>{error}</span></div>}

            <form className="composer" onSubmit={sendMessage}>
              <textarea
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask your documents…"
                rows={1}
                disabled={isLoading || messagesLoading}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    if (question.trim() && !isLoading && !messagesLoading) event.currentTarget.form?.requestSubmit();
                  }
                }}
              />
              <button type="submit" className="send-button" disabled={!question.trim() || isLoading || messagesLoading}>
                {isLoading ? "…" : "Send ↗"}
              </button>
            </form>
            <p className="composer-hint">{documents.length === 0 ? "Upload a document to ground your answers in evidence." : "Enter to send · Shift + Enter for a new line · Answers link to sources."}</p>
            {copyNotice && <div className="copy-notice" role="status">{copyNotice}<button type="button" onClick={() => setCopyNotice(null)}>×</button></div>}
          </section>

          <aside className={`evidence-panel${evidencePanelCollapsed ? " evidence-panel-collapsed" : ""}`} aria-label="Retrieved evidence">
            <div className="evidence-panel-header">
              <div>
                <span className="chat-context-label">CURRENT EVIDENCE</span>
                <h2>Retrieved sources</h2>
                <p>{evidenceSources.length} {evidenceSources.length === 1 ? "source" : "sources"} from the latest answer</p>
              </div>
              <button
                type="button"
                className="evidence-toggle"
                onClick={() => setEvidencePanelCollapsed((current) => !current)}
                aria-expanded={!evidencePanelCollapsed}
                aria-label={evidencePanelCollapsed ? "Expand retrieved sources panel" : "Collapse retrieved sources panel"}
                title={evidencePanelCollapsed ? "Expand sources" : "Collapse sources"}
              >
                {evidencePanelCollapsed ? "›" : "‹"}
              </button>
            </div>
            {!evidencePanelCollapsed && (
              <div className="evidence-list">
                {sourcePreviewLoading && <p role="status">Loading source passage…</p>}
                {sourcePreviewError && <p role="alert">{sourcePreviewError}</p>}
                {sourcePreview && (
                  <section className="source-preview-card" aria-label="Cited source text">
                    <strong>{sourcePreview.filename} — chunk {sourcePreview.chunk_id}</strong>
                    <button type="button" onClick={() => void copyText(sourcePreview.text)}>Copy passage</button><button type="button" onClick={() => setSourcePreview(null)} aria-label="Close source passage">Close</button>
                    <p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{sourcePreview.text}</p>
                  </section>
                )}
                {evidenceSources.length === 0 ? (
                  <div className="evidence-empty">
                    <span className="evidence-empty-mark">↗</span>
                    <strong>No retrieved sources yet</strong>
                    <p>Sources returned with an answer will appear here.</p>
                  </div>
                ) : evidenceSources.map((source, index) => (
                  <article className="evidence-card" key={`${source.document_id}-${source.chunk_id}-${index}`}>
                    <div className="evidence-card-top">
                      <span className="evidence-document-mark">{source.filename.split(".").pop()?.toUpperCase() ?? "DOC"}</span>
                      <span className="evidence-score">{source.score.toFixed(2)}</span>
                    </div>
                    <strong className="evidence-filename" title={source.filename}>{source.filename}</strong>
                    <p className="evidence-citation">{source.citation}</p>
                    <div className="evidence-card-footer">Chunk {source.chunk_id}</div>
                    <button
                      type="button"
                      className="text-button"
                      onClick={() => void previewSource(source)}
                      aria-label={`Read cited passage ${source.citation} from ${source.filename}`}
                    >
                      Read source passage
                    </button>
                  </article>
                ))}
              </div>
            )}
          </aside>
        </div>
      </section>
    </main>
  );
}

export default function Home() {
  const [authenticated, setAuthenticated] = useState<boolean | null>(null);
  const [password, setPassword] = useState("");
  const [loginError, setLoginError] = useState<string | null>(null);
  const [loggingIn, setLoggingIn] = useState(false);

  useEffect(() => {
    let active = true;
    fetch("/api/auth/session", { cache: "no-store" })
      .then((response) => response.json())
      .then((result: { authenticated?: boolean }) => {
        if (active) setAuthenticated(result.authenticated === true);
      })
      .catch(() => { if (active) setAuthenticated(false); });
    return () => { active = false; };
  }, []);

  async function signIn(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoggingIn(true);
    setLoginError(null);
    try {
      const response = await fetch("/api/auth/login", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ password }),
      });
      if (!response.ok) throw await responseError(response, "Sign in failed.");
      setPassword("");
      setAuthenticated(true);
    } catch (error) {
      setLoginError(error instanceof Error ? error.message : "Sign in failed.");
    } finally {
      setLoggingIn(false);
    }
  }

  async function signOut() {
    await fetch("/api/auth/logout", { method: "POST" });
    setAuthenticated(false);
  }

  if (authenticated === null) return (
    <main className="auth-loading" role="status" aria-live="polite">
      <span className="brand-mark" aria-hidden="true">◆</span> Opening your private workspace…
    </main>
  );
  if (!authenticated) return (
    <main className="auth-shell">
      <header className="auth-header">
        <div className="brand-lockup"><span className="brand-mark" aria-hidden="true">◆</span><span>localrag</span></div>
        <span className="auth-header-label"><span className="status-dot" /> SELF-HOSTED KNOWLEDGE WORKSPACE</span>
      </header>
      <div className="auth-layout">
        <section className="auth-story" aria-label="About LocalRAG">
          <span className="auth-kicker">PRIVATE BY DESIGN · POWERED BY YOUR DATA</span>
          <h1>Your documents.<br />Your answers.<br /><em>Your control.</em></h1>
          <p>Turn scattered files into searchable knowledge. Ask better questions, uncover what matters, and see exactly where every answer came from.</p>
          <div className="auth-steps" aria-label="How it works">
            <div><span>01</span><div><strong>Bring your knowledge</strong><p>Upload PDFs, Word documents, Markdown, and text files.</p></div></div>
            <div><span>02</span><div><strong>Ask naturally</strong><p>Explore your library through a conversation.</p></div></div>
            <div><span>03</span><div><strong>Verify every insight</strong><p>Open the passages behind cited answers.</p></div></div>
          </div>
        </section>
        <section className="auth-card" aria-labelledby="auth-title">
          <div className="auth-card-icon" aria-hidden="true">✦</div>
          <p className="eyebrow">WELCOME BACK</p>
          <h2 id="auth-title">Sign in to LocalRAG</h2>
          <p className="auth-card-description">Your private knowledge workspace is ready when you are.</p>
          <form onSubmit={(event) => void signIn(event)}>
            <label htmlFor="localrag-password">Administrator password</label>
            <input
              id="localrag-password"
              type="password"
              autoComplete="current-password"
              placeholder="Enter your administrator password"
              required
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              aria-invalid={Boolean(loginError)}
              aria-describedby={loginError ? "auth-error" : undefined}
            />
            {loginError && <div id="auth-error" className="auth-error" role="alert">{loginError}</div>}
            <button type="submit" className="auth-submit" disabled={loggingIn || !password.trim()}>
              {loggingIn ? "Verifying…" : "Enter workspace"} <span aria-hidden="true">↗</span>
            </button>
          </form>
          <div className="auth-card-foot"><span className="status-dot" /> Runs on your own infrastructure. No cloud account required.</div>
        </section>
      </div>
      <footer className="auth-footer"><span>LOCALRAG · LOCAL-FIRST INTELLIGENCE</span><span>Ollama / Qdrant / FastAPI</span></footer>
    </main>
  );
  return <Workspace onLogout={() => void signOut()} />;
}
