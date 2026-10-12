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

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

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

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
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
  const [conversationActionId, setConversationActionId] = useState<string | null>(null);

  const [documents, setDocuments] = useState<Document[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null);
  const [documentError, setDocumentError] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);
  const initialLoadStartedRef = useRef(false);
  const messageLoadRequestRef = useRef(0);

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
    // eslint-disable-next-line react-hooks/set-state-in-effect
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
    setUploading(true);
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
      console.log(`Indexed ${data.filename} (${data.chunks} chunks)`);
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
  const visibleConversations = conversations.filter((item) => item.is_archived === showArchived);
  const conversationControlsDisabled = isLoading || messagesLoading || conversationActionId !== null;

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="app-header">
          <div>
            <p className="eyebrow">LOCALRAG</p>
            <h1>Ask your documents.</h1>
            <p className="subtitle">Private, local document retrieval powered by Ollama and Qdrant.</p>
          </div>
          <div className="status-pill"><span className="status-dot" />Local</div>
        </header>

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
                    <strong>{showArchived ? "No archived chats" : "No conversations yet"}</strong>
                    <span>{showArchived ? "Archived chats will appear here." : "Start a new chat to keep your questions and answers."}</span>
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
              <div className="document-list">
                {documentsLoading ? (
                  <div className="documents-empty">Loading documents…</div>
                ) : documents.length === 0 ? (
                  <div className="documents-empty">
                    <div className="documents-empty-icon">+</div>
                    <strong>No documents yet</strong>
                    <span>Upload a document to start asking questions.</span>
                  </div>
                ) : documents.map((document) => (
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
              <div className="upload-hint">Supported: PDF, DOCX, TXT, MD</div>
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
                <div className="empty-icon">⌕</div>
                <h2>Ask something about your documents</h2>
                <p>LocalRAG retrieves relevant document chunks and uses them as context for the answer. Your questions and answers are saved in the selected conversation.</p>
                <div className="suggestions">
                  <button type="button" onClick={() => setQuestion("What does LocalRAG use for vector storage?")}>What does LocalRAG use for vector storage?</button>
                  <button type="button" onClick={() => setQuestion("How does LocalRAG retrieve relevant documents?")}>How does document retrieval work?</button>
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
                {isLoading ? "…" : "Send"}
              </button>
            </form>
            <p className="composer-hint">Press Enter to send · Shift + Enter for a new line</p>
          </section>
        </div>
      </section>
    </main>
  );
}
