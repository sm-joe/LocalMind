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
  const [activeView, setActiveView] = useState<"library" | "research" | "history">("library");
  const [fileTypeFilter, setFileTypeFilter] = useState("all");
  const [selectedDocumentId, setSelectedDocumentId] = useState<string | null>(null);
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

  async function openConversation(conversationId: string, navigate = true) {
    const requestId = ++messageLoadRequestRef.current;
    setActiveConversationId(conversationId);
    if (navigate) setActiveView("research");
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
        await openConversation(mostRecentActive.id, false);
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
      setActiveView("research");
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
    setActiveView("research");

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

      const assistantId = `streaming-${crypto.randomUUID()}`;
      setMessages((current) => [
        ...current,
        { id: assistantId, role: "assistant", content: "", sources: [], status: "pending" },
      ]);
      const response = await fetch(`${API_URL}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Idempotency-Key": crypto.randomUUID(),
        },
        body: JSON.stringify({ question: trimmedQuestion, top_k: 3, conversation_id: conversationId, ...(selectedDocumentId ? { document_id: selectedDocumentId } : {}) }),
      });
      if (!response.ok) {
        throw await responseError(response, "Unable to start document answering.");
      }
      if (!response.body) throw new Error("The backend did not provide an answer stream.");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let completed = false;
      const consumeEvent = (payload: string) => {
        if (!payload.startsWith("data: ")) return;
        const event = JSON.parse(payload.slice(6)) as {
          type: "chunk" | "done" | "error" | "status";
          text?: string;
          detail?: string;
          result?: ChatResponse;
        };
        if (event.type === "chunk" && event.text) {
          const token = event.text;
          setMessages((current) => current.map((message) =>
            message.id === assistantId
              ? { ...message, content: message.content + token }
              : message,
          ));
        } else if (event.type === "done" && event.result) {
          completed = true;
          const data = event.result;
          setMessages((current) => [
            ...current.filter((message) => message.id !== optimisticId && message.id !== assistantId),
            {
              id: data.user_message_id ?? optimisticId,
              role: "user" as const,
              content: trimmedQuestion,
              status: "complete" as const,
            },
            {
              id: data.assistant_message_id ?? assistantId,
              role: "assistant" as const,
              content: data.answer,
              sources: data.sources,
              status: "complete" as const,
            },
          ]);
        } else if (event.type === "error") {
          throw new Error(event.detail ?? "The local model could not answer.");
        }
      };
      while (true) {
        let next: ReadableStreamReadResult<Uint8Array>;
        try {
          next = await reader.read();
        } catch {
          // Streaming fetch can already have returned HTTP 200 when the
          // proxy, Ollama, or API disconnects. A browser TypeError otherwise
          // surfaces as an unhelpful generic "Network error".
          throw new Error(
            "The answer stream disconnected before completion. " +
            "Check the web, api, and ollama container logs; indexed documents are unchanged.",
          );
        }
        const { done, value } = next;
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";
        for (const frame of frames) consumeEvent(frame.trim());
      }
      buffer += decoder.decode();
      if (buffer.trim()) consumeEvent(buffer.trim());
      if (!completed) throw new Error("The model disconnected without completing its answer.");
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
      let data = (await response.json()) as UploadResponse;
      if (data.status === "already_indexed" && window.confirm(
        "This file is already indexed. Rebuild its search index from the original document? " +
        "This can recover content missed by older DOCX parsing. The existing index stays available until replacement succeeds."
      )) {
        const rebuild = await fetch(`${API_URL}/documents/upload?reindex=true`, {
          method: "POST",
          body: formData,
        });
        if (!rebuild.ok) throw await responseError(rebuild, "Document reindexing failed.");
        data = (await rebuild.json()) as UploadResponse;
      }
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
    item.filename.toLowerCase().includes(documentSearch.trim().toLowerCase()) &&
    (fileTypeFilter === "all" || item.filename.toLowerCase().endsWith("." + fileTypeFilter)),
  );
  const indexedChunks = documents.reduce((total, item) => total + item.chunks, 0);
  const selectedDocument = documents.find((item) => item.document_id === selectedDocumentId);
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
    <main className="lm-app">
      <aside className="lm-sidebar" aria-label="Main navigation">
        <div className="lm-brand"><span className="lm-brand-symbol" aria-hidden="true">m.</span><span>LocalMind</span></div>
        <div className="lm-workspace-select"><span className="lm-workspace-avatar">L</span><div><strong>Personal Workspace</strong><small>Private Knowledge Base</small></div></div>
        <div className="lm-nav-label">WORKSPACE</div>
        <nav className="lm-navigation">
          <button type="button" className={activeView === "library" ? "lm-nav-link active" : "lm-nav-link"} aria-current={activeView === "library" ? "page" : undefined} onClick={() => setActiveView("library")}>
            <span aria-hidden="true">▦</span> Documents <small>{documents.length}</small>
          </button>
          <button type="button" className={activeView === "research" ? "lm-nav-link active" : "lm-nav-link"} aria-current={activeView === "research" ? "page" : undefined} onClick={() => setActiveView("research")}>
            <span aria-hidden="true">⌕</span> Research
          </button>
          <button type="button" className={activeView === "history" ? "lm-nav-link active" : "lm-nav-link"} aria-current={activeView === "history" ? "page" : undefined} onClick={() => setActiveView("history")}>
            <span aria-hidden="true">▤</span> Conversations <small>{conversations.filter((item) => !item.is_archived).length}</small>
          </button>
        </nav>
        <div className="lm-nav-label lm-quick-heading">QUICK ACTIONS</div>
        <button type="button" className="lm-sidebar-action" onClick={openFilePicker} disabled={uploading}>+ Add Documents</button>
        <button type="button" className="lm-sidebar-action muted" disabled={conversationControlsDisabled} onClick={() => void createConversation()}>+ New Research Thread</button>
        <div className="lm-sidebar-bottom"><span className="lm-local-indicator" /> Local Workspace <small>Your Files Stay In Your Environment</small></div>
      </aside>

      <section className="lm-main">
        <header className="lm-topbar">
          <div className="lm-breadcrumb">Workspace <span>/</span> <strong>{activeView === "library" ? "Documents" : activeView === "research" ? "Research" : "Conversations"}</strong></div>
          <div className="lm-header-right"><span className="lm-topbar-privacy">● Private &amp; Local</span><button type="button" className="lm-signout" onClick={onLogout}>Sign Out</button></div>
        </header>
        <div className="lm-content">
          {activeView === "library" && (
            <section className="lm-view lm-library" aria-label="Document library">
              <div className="lm-page-header">
                <div><p className="lm-overline">YOUR WORKSPACE / FILE MANAGEMENT</p><h1>Documents</h1><p>Keep Your Knowledge Organized, Searchable, And Ready To Explore.</p></div>
                <button className="lm-primary-button" type="button" onClick={openFilePicker} disabled={uploading}>{uploading ? "Indexing..." : "+ Upload Documents"}</button>
              </div>
              <div className="lm-stats">
                <div><span>Files In Your Library</span><strong>{documents.length}</strong><small>Supported Documents</small></div>
                <div><span>Searchable Passages</span><strong>{indexedChunks}</strong><small>Indexed Chunks</small></div>
                <div><span>Research Conversations</span><strong>{conversations.filter((item) => !item.is_archived).length}</strong><small>Saved Threads</small></div>
              </div>
              <div className={"lm-upload-zone" + (dragActive ? " is-dragging" : "")} onDragOver={(event) => { event.preventDefault(); if (!uploading) setDragActive(true); }} onDragLeave={(event) => { event.preventDefault(); setDragActive(false); }} onDrop={(event) => { event.preventDefault(); setDragActive(false); if (!uploading && event.dataTransfer.files[0]) void uploadDocument(event.dataTransfer.files[0]); }}>
                <div className="lm-upload-icon" aria-hidden="true">↑</div>
                <div><strong>{dragActive ? "Release To Add Your Document" : "Drag And Drop Files Here"}</strong><p>PDF, Word, Markdown Or Text · Up To 10 MB Per File</p></div>
                <button type="button" onClick={openFilePicker} disabled={uploading} className="lm-outline-button">{uploading ? "Indexing..." : "Browse Files"}</button>
              </div>
              {documentError && <div className="lm-error" role="alert">{documentError}</div>}
              {uploadNotice && <div className="lm-notice" role="status">{uploadNotice}<button type="button" aria-label="Dismiss upload notice" onClick={() => setUploadNotice(null)}>×</button></div>}
              <div className="lm-files-panel">
                <div className="lm-section-header"><div><h2>Your Files</h2><p>{visibleDocuments.length} of {documents.length} documents</p></div><label className="lm-search"><span aria-hidden="true">⌕</span><input type="search" placeholder="Search File Names" aria-label="Search File Names" value={documentSearch} onChange={(event) => setDocumentSearch(event.target.value)} /></label></div>
                <div className="lm-filters" aria-label="Filter File Type">
                  {["all", "pdf", "docx", "txt", "md"].map((type) => <button type="button" key={type} aria-pressed={fileTypeFilter === type} className={fileTypeFilter === type ? "active" : ""} onClick={() => setFileTypeFilter(type)}>{type === "all" ? "All Files" : type.toUpperCase()}</button>)}
                </div>
                <div className="lm-file-table" role="table" aria-label="Indexed Documents">
                  <div className="lm-file-row lm-file-thead" role="row"><span role="columnheader">NAME</span><span role="columnheader">TYPE</span><span role="columnheader">INDEXED CONTENT</span><span role="columnheader">ACTIONS</span></div>
                  {documentsLoading ? <div className="lm-table-empty" role="status">Loading Your Library...</div> :
                   visibleDocuments.length === 0 ? <div className="lm-table-empty"><strong>{documents.length === 0 ? "Your Library Is Empty" : "No Matching Files"}</strong><p>{documents.length === 0 ? "Add A Document Above To Start Exploring." : "Try A Different Search Or File Type."}</p></div> :
                   visibleDocuments.map((document) => (
                     <div className="lm-file-row" role="row" key={document.document_id}>
                       <div className="lm-file-name" role="cell"><span className="lm-file-icon">{document.filename.split(".").pop()?.toUpperCase().slice(0, 4) || "FILE"}</span><div><strong title={document.filename}>{document.filename}</strong><small>Ready For Questions</small></div></div>
                       <span className="lm-type" role="cell">{document.filename.split(".").pop()?.toUpperCase() || "FILE"}</span>
                       <span className="lm-chunks" role="cell">{document.chunks} passages</span>
                       <div className="lm-file-actions" role="cell">
                         <button type="button" onClick={() => { setSelectedDocumentId(document.document_id); setActiveView("research"); setQuestion(""); }} aria-label={"Ask about " + document.filename}>Ask Questions ↗</button>
                         <button type="button" className="lm-delete" onClick={() => void deleteDocument(document.document_id)} disabled={deletingDocumentId === document.document_id} aria-label={"Delete " + document.filename}>{deletingDocumentId === document.document_id ? "..." : "Delete"}</button>
                       </div>
                     </div>
                   ))}
                </div>
              </div>
            </section>
          )}

          {activeView === "history" && (
            <section className="lm-view lm-history" aria-label="Conversation History">
              <div className="lm-page-header"><div><p className="lm-overline">YOUR RESEARCH ARCHIVE</p><h1>Conversations</h1><p>Find, Continue And Organize Your Saved Research Threads.</p></div><button type="button" className="lm-primary-button" disabled={conversationControlsDisabled} onClick={() => void createConversation()}>+ New Conversation</button></div>
              <div className="lm-history-panel">
                <div className="lm-section-header"><div><h2>{showArchived ? "Archived Conversations" : "Recent Conversations"}</h2><p>{visibleConversations.length} threads</p></div><div className="lm-history-tools"><label className="lm-search"><span aria-hidden="true">⌕</span><input type="search" aria-label="Find A Conversation" placeholder="Search Conversations" value={conversationSearch} onChange={(event) => setConversationSearch(event.target.value)} /></label><button type="button" className="lm-outline-button" onClick={() => setShowArchived((v) => !v)}>{showArchived ? "Show Active" : "Archived"}</button></div></div>
                {conversationError && <div className="lm-error" role="alert">{conversationError}</div>}
                {conversationsLoading ? <div className="lm-table-empty">Loading Conversations...</div> : visibleConversations.length === 0 ? <div className="lm-table-empty"><strong>No Matching Conversations</strong><p>Your Conversations Will Appear Here.</p></div> : visibleConversations.map((conversation) => (
                  <article className="lm-thread" key={conversation.id}>
                    <button type="button" className="lm-thread-open" disabled={conversationControlsDisabled} onClick={() => void openConversation(conversation.id)}>
                      <span className="lm-thread-icon">▤</span><span><strong>{conversation.title}</strong><small>{conversation.message_count} messages · {new Date(conversation.updated_at).toLocaleDateString()}</small></span><span className="lm-thread-arrow">↗</span>
                    </button>
                    <div className="lm-thread-actions">
                      {!conversation.is_archived && <button type="button" disabled={conversationControlsDisabled} onClick={() => void renameConversation(conversation)}>Rename</button>}
                      <button type="button" disabled={conversationControlsDisabled} onClick={() => void toggleArchive(conversation)}>{conversation.is_archived ? "Restore" : "Archive"}</button>
                      <button type="button" className="lm-delete" disabled={conversationControlsDisabled} onClick={() => void deleteConversation(conversation)}>Delete</button>
                    </div>
                  </article>
                ))}
                {!conversationsLoading && conversationNextCursor && <button className="lm-load-more" type="button" onClick={() => void loadMoreConversations()} disabled={loadingMoreConversations || conversationControlsDisabled}>{loadingMoreConversations ? "Loading..." : "Load More Conversations"}</button>}
              </div>
            </section>
          )}

          {activeView === "research" && (
            <section className="lm-view lm-research" aria-label="Document Research">
              <div className="lm-page-header lm-research-header"><div><p className="lm-overline">ASK &amp; VERIFY / RESEARCH</p><h1>Research</h1><p>Explore Your Material And Inspect The Passages Behind Each Answer.</p></div><button type="button" className="lm-outline-button" disabled={conversationControlsDisabled} onClick={() => void createConversation()}>+ New Thread</button></div>
              <div className="lm-scope-toolbar"><label htmlFor="lm-document-scope">Source Collection</label><select id="lm-document-scope" value={selectedDocumentId ?? ""} onChange={(event) => setSelectedDocumentId(event.target.value || null)}><option value="">All Documents</option>{documents.map((doc) => <option value={doc.document_id} key={doc.document_id}>{doc.filename}</option>)}</select><span>{selectedDocument ? "Questions Are Limited To This Document" : "Search Across Your Indexed Library"}</span></div>
              <div className="lm-research-grid">
                <section className="lm-answer-panel" aria-label="Question And Answer">
                  <div className="lm-answer-header"><div><small>ACTIVE THREAD</small><h2>{activeConversation?.title ?? "New Research"}</h2></div><span>{messages.length} messages</span></div>
                  {messagesLoading ? <div className="lm-answer-empty" role="status">Opening Conversation...</div> :
                    messages.length === 0 ? (
                      <div className="lm-answer-empty"><span className="lm-answer-empty-icon" aria-hidden="true">⌕</span><h2>Start With A Question.</h2><p>Summarize A File, Explore A Concept Or Compare Two Ideas. The Source Passages Appear Alongside Each Answer.</p><div className="lm-starter-prompts"><button type="button" onClick={() => setQuestion("Summarize the document.")}>Summarize The Document ↗</button><button type="button" onClick={() => setQuestion("What are the key topics covered?")}>Find Key Topics ↗</button></div></div>
                    ) : <div className="lm-answer-messages">{messages.map((message) => <article className={"lm-answer-message " + (message.role === "user" ? "is-user" : "is-assistant")} key={message.id}><div className="lm-message-author">{message.role === "user" ? "YOU ASKED" : "LOCALMIND"}</div><div className="lm-message-text">{message.status === "failed" && !message.content ? "This Answer Could Not Be Generated." : message.content}</div>{message.role === "assistant" && message.status === "complete" && <button type="button" className="lm-copy" onClick={() => void copyText(message.content)}>Copy Answer</button>}{message.status === "failed" && <span className="lm-message-failed">Generation Failed</span>}</article>)}{isLoading && <div className="lm-thinking" role="status">Researching Your Documents...</div>}<div ref={messagesEndRef} aria-hidden="true" /></div>}
                  {error && <div className="lm-error" role="alert">{error}</div>}
                  <form className="lm-composer" onSubmit={sendMessage}><textarea aria-label="Ask A Question About Your Documents" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask A Question About Your Documents..." rows={2} disabled={isLoading || messagesLoading} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); if (question.trim() && !isLoading && !messagesLoading) event.currentTarget.form?.requestSubmit(); } }} /><div><small>Enter To Ask · Shift + Enter For A New Line</small><button className="lm-primary-button" type="submit" disabled={!question.trim() || isLoading || messagesLoading}>{isLoading ? "Working..." : "Ask Question ↗"}</button></div></form>
                  {copyNotice && <div className="lm-notice" role="status">{copyNotice}<button type="button" onClick={() => setCopyNotice(null)}>×</button></div>}
                </section>
                <aside className="lm-evidence-panel" aria-label="Source Evidence">
                  <div className="lm-evidence-header"><div><small>REFERENCES</small><h2>Source Evidence</h2><p>{evidenceSources.length} Passages From The Latest Answer</p></div><button type="button" aria-label={evidencePanelCollapsed ? "Expand Sources" : "Collapse Sources"} aria-expanded={!evidencePanelCollapsed} onClick={() => setEvidencePanelCollapsed((v) => !v)}>{evidencePanelCollapsed ? "+" : "−"}</button></div>
                  {!evidencePanelCollapsed && <div className="lm-evidence-body">
                    {sourcePreviewLoading && <p role="status">Loading Source Passage...</p>}
                    {sourcePreviewError && <p role="alert" className="lm-error">{sourcePreviewError}</p>}
                    {sourcePreview && <section className="lm-source-preview"><div><strong>{sourcePreview.filename}</strong><button type="button" onClick={() => setSourcePreview(null)}>Close</button></div><small>PASSAGE {sourcePreview.chunk_id}</small><p>{sourcePreview.text}</p><button type="button" className="lm-outline-button" onClick={() => void copyText(sourcePreview.text)}>Copy Passage</button></section>}
                    {evidenceSources.length === 0 ? <div className="lm-evidence-empty"><span aria-hidden="true">▤</span><strong>Evidence, Not Guesswork.</strong><p>Retrieved Source Passages Will Appear Here After Your Question.</p></div> :
                    evidenceSources.map((source, index) => <article className="lm-evidence-item" key={source.document_id + "-" + source.chunk_id + "-" + index}><span className="lm-citation">{source.citation}</span><div><strong title={source.filename}>{source.filename}</strong><small>Passage {source.chunk_id}</small><button type="button" onClick={() => void previewSource(source)}>Read Original Passage ↗</button></div></article>)}
                  </div>}
                </aside>
              </div>
            </section>
          )}
        </div>
        <footer className="lm-footer"><span>LocalMind · Private Knowledge Workspace</span><span>Documents First. Answers With Context.</span></footer>
        <input ref={fileInputRef} type="file" className="lm-hidden-file" accept=".pdf,.docx,.txt,.md" onChange={handleFileChange} />
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
      <span className="brand-mark" aria-hidden="true">M<span className="brand-mark-detail">/</span></span> Opening LocalMind…
    </main>
  );
  if (!authenticated) return (
    <main className="auth-shell auth-reading-desk">
      <div className="auth-page-inner">
        <header className="auth-header">
          <div className="brand-lockup">
            <span className="brand-mark" aria-hidden="true">M<span className="brand-mark-detail">/</span></span>
            <span>LocalMind<span className="brand-mark-subtle"> / The Reading Desk</span></span>
          </div>
          <span className="auth-header-label"><span className="status-dot" /> PRIVATE. LOCAL. YOURS.</span>
        </header>

        <div className="auth-layout">
          <section className="auth-story" aria-label="About LocalMind">
            <div className="auth-editorial-header">
              <p className="auth-kicker"><span>01 /</span> A HOME FOR YOUR KNOWLEDGE</p>
              <h1>Less Searching.<br /><em>More Finding.</em></h1>
              <p className="auth-story-description">
                A Calm Place To Read Across Your Documents, Ask Better Questions
                And Follow Every Answer Back To Its Source.
              </p>
            </div>
            <div className="auth-document-preview" aria-label="Illustration of the LocalMind document reading workflow">
              <div className="auth-preview-head">
                <span>THE READING DESK</span>
                <span>ANATOMY OF AN ANSWER / 01</span>
              </div>
              <div className="auth-preview-paper">
                <div className="auth-preview-paper-heading">
                  <span className="auth-preview-file-icon" aria-hidden="true">D</span>
                  <div><span>YOUR LIBRARY</span><strong>Notes On The Things That Matter.</strong></div>
                  <span className="auth-preview-page-number">01—03</span>
                </div>
                <p className="auth-preview-label">A QUESTION WORTH ASKING</p>
                <h2>What Does The Document<br />Actually Say?</h2>
                <div className="auth-preview-rules" aria-hidden="true">
                  <span /><span /><span />
                </div>
                <div className="auth-preview-quote">
                  <span className="auth-preview-quote-mark" aria-hidden="true">“</span>
                  <p>Answers Grounded In Your Documents, With The Passages Beside Them.</p>
                  <span className="auth-preview-citation">SOURCE / 001</span>
                </div>
              </div>
              <div className="auth-preview-bottom"><span>READ</span><span>QUESTION</span><span>TRACE</span></div>
            </div>
            <div className="auth-steps" aria-label="How LocalMind works">
              <div><span>01</span><div><strong>Collect</strong><p>PDF, Word, Markdown And Plain Text.</p></div></div>
              <div><span>02</span><div><strong>Explore</strong><p>Ask Questions and connect ideas.</p></div></div>
              <div><span>03</span><div><strong>Verify</strong><p>Inspect The Original Passages.</p></div></div>
            </div>
          </section>

          <section className="auth-signin-zone" aria-labelledby="auth-title">
            <div className="auth-signin-topline"><span>YOUR PRIVATE WORKSPACE</span><span>ACCESS / 01</span></div>
            <div className="auth-card">
              <div className="auth-card-icon" aria-hidden="true">M<span>/</span></div>
              <p className="eyebrow">WELCOME BACK</p>
              <h2 id="auth-title">Make Yourself<br /><em>At Home.</em></h2>
              <p className="auth-card-description">
                Sign In To Continue Your Research, Documents And Conversations.
              </p>
              <form onSubmit={(event) => void signIn(event)}>
                <label htmlFor="localmind-password">Administrator Password</label>
                <input
                  id="localmind-password"
                  type="password"
                  autoComplete="current-password"
                  placeholder="Enter Your Password"
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  aria-invalid={Boolean(loginError)}
                  aria-describedby={loginError ? "auth-error" : undefined}
                />
                {loginError && <div id="auth-error" className="auth-error" role="alert">{loginError}</div>}
                <button type="submit" className="auth-submit" disabled={loggingIn || !password.trim()}>
                  <span>{loggingIn ? "Verifying Access…" : "Open Your Desk"}</span><span aria-hidden="true">↗</span>
                </button>
              </form>
              <div className="auth-card-foot">
                <span className="auth-privacy-mark" aria-hidden="true">●</span>
                Self-hosted. Your Files Stay In Your Environment.
              </div>
            </div>
            <p className="auth-signin-note">A PERSONAL SPACE FOR THOUGHTFUL WORK.</p>
          </section>
        </div>

        <footer className="auth-footer">
          <span>LOCALMIND / THE READING DESK</span>
          <span>DOCUMENTS IN. CLARITY OUT.</span>
        </footer>
      </div>
    </main>
  );
  return <Workspace onLogout={() => void signOut()} />;
}
