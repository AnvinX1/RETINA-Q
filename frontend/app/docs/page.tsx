"use client";

import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { docsContent, type DocEntry } from "@/lib/docs-content";
import {
    Eye, Database, ScanLine, Atom, BrainCircuit, Layers,
    Grid3x3, Server, Monitor, Lightbulb, Container, RefreshCw,
    ChevronLeft, ChevronRight, Search, BookOpen, List, ArrowUp,
} from "lucide-react";

const iconMap: Record<string, React.ElementType> = {
    Eye, Database, ScanLine, Atom, BrainCircuit, Layers,
    Grid3x3, Server, Monitor, Lightbulb, Container, RefreshCw,
    BookOpen,
};

export default function DocsPage() {
    const [activeDoc, setActiveDoc] = useState(0);
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [searchQuery, setSearchQuery] = useState("");
    const [showToc, setShowToc] = useState(false);
    const [showScrollTop, setShowScrollTop] = useState(false);
    const contentRef = useRef<HTMLDivElement>(null);

    const doc = docsContent[activeDoc];

    // Filter docs by search
    const filteredDocs = useMemo(() => {
        if (!searchQuery.trim()) return docsContent;
        const q = searchQuery.toLowerCase();
        return docsContent.filter(
            (d) =>
                d.title.toLowerCase().includes(q) ||
                d.short.toLowerCase().includes(q) ||
                d.content.toLowerCase().includes(q)
        );
    }, [searchQuery]);

    // Extract headings for TOC
    const headings = useMemo(() => {
        const lines = doc.content.split("\n");
        const results: { level: number; text: string; id: string }[] = [];
        for (const line of lines) {
            const match = line.match(/^(#{1,3})\s+(.+)/);
            if (match) {
                const level = match[1].length;
                const text = match[2].replace(/\*\*/g, "");
                const id = text
                    .toLowerCase()
                    .replace(/[^a-z0-9\s-]/g, "")
                    .replace(/\s+/g, "-")
                    .slice(0, 60);
                results.push({ level, text, id });
            }
        }
        return results;
    }, [doc]);

    // Scroll to top on doc change
    useEffect(() => {
        contentRef.current?.scrollTo({ top: 0, behavior: "smooth" });
    }, [activeDoc]);

    // Show scroll-to-top button
    useEffect(() => {
        const el = contentRef.current;
        if (!el) return;
        const handler = () => setShowScrollTop(el.scrollTop > 400);
        el.addEventListener("scroll", handler, { passive: true });
        return () => el.removeEventListener("scroll", handler);
    }, []);

    const scrollToHeading = useCallback((id: string) => {
        const el = contentRef.current?.querySelector(`#${CSS.escape(id)}`);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
        setShowToc(false);
    }, []);

    const goNext = () => setActiveDoc((p) => Math.min(p + 1, docsContent.length - 1));
    const goPrev = () => setActiveDoc((p) => Math.max(p - 1, 0));

    return (
        <div className="max-w-[1440px] mx-auto flex h-[calc(100vh-7rem)]">
            {/* Sidebar */}
            <aside
                className={`shrink-0 border-r border-border bg-white transition-all duration-200 overflow-hidden flex flex-col ${
                    sidebarOpen ? "w-72" : "w-0 border-r-0"
                }`}
            >
                <div className="p-4 border-b border-border">
                    <div className="flex items-center gap-2 mb-3">
                        <BookOpen size={14} className="text-accent" />
                        <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-muted-foreground">
                            Documentation
                        </span>
                    </div>
                    <div className="relative">
                        <Search size={12} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search docs..."
                            className="w-full pl-7 pr-3 py-1.5 text-[11px] font-mono bg-muted border border-border rounded focus:outline-none focus:border-foreground transition-colors"
                        />
                    </div>
                </div>
                <nav className="flex-1 overflow-y-auto py-2">
                    {filteredDocs.map((d) => {
                        const Icon = iconMap[d.icon] || BookOpen;
                        const isActive = docsContent[activeDoc].id === d.id;
                        return (
                            <button
                                key={d.id}
                                onClick={() => {
                                    setActiveDoc(docsContent.findIndex((x) => x.id === d.id));
                                    setSearchQuery("");
                                }}
                                className={`w-full text-left px-4 py-2.5 flex items-start gap-3 transition-colors ${
                                    isActive
                                        ? "bg-muted border-l-2 border-foreground"
                                        : "border-l-2 border-transparent hover:bg-muted/50"
                                }`}
                            >
                                <Icon
                                    size={13}
                                    className={`mt-0.5 shrink-0 ${isActive ? "text-foreground" : "text-muted-foreground"}`}
                                />
                                <div className="min-w-0">
                                    <p
                                        className={`text-[11px] font-mono leading-tight truncate ${
                                            isActive ? "font-bold text-foreground" : "text-foreground"
                                        }`}
                                    >
                                        <span className="text-muted-foreground mr-1">{String(d.id).padStart(2, "0")}.</span>
                                        {d.title.replace(/^\d+\s*—\s*/, "")}
                                    </p>
                                    <p className="text-[9px] font-mono text-muted-foreground mt-0.5 truncate">
                                        {d.short}
                                    </p>
                                </div>
                            </button>
                        );
                    })}
                </nav>
                <div className="p-3 border-t border-border text-[9px] font-mono text-muted-foreground text-center">
                    {docsContent.length} documents &middot; Explainability Series
                </div>
            </aside>

            {/* Main Content */}
            <div className="flex-1 flex flex-col min-w-0">
                {/* Toolbar */}
                <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-white shrink-0">
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => setSidebarOpen(!sidebarOpen)}
                            className="p-1.5 hover:bg-muted rounded transition-colors"
                            title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
                        >
                            {sidebarOpen ? <ChevronLeft size={14} /> : <List size={14} />}
                        </button>
                        <div className="flex items-center gap-1.5 text-[10px] font-mono text-muted-foreground">
                            <span className="text-accent font-bold">$</span>
                            <span>cat</span>
                            <span className="text-foreground font-bold">
                                {String(doc.id).padStart(2, "0")}_{doc.slug.toUpperCase()}.md
                            </span>
                        </div>
                    </div>
                    <div className="flex items-center gap-1">
                        <button
                            onClick={() => setShowToc(!showToc)}
                            className="px-2 py-1 text-[10px] font-mono text-muted-foreground hover:text-foreground hover:bg-muted rounded transition-colors border border-border"
                        >
                            TOC
                        </button>
                        <button
                            onClick={goPrev}
                            disabled={activeDoc === 0}
                            className="p-1.5 hover:bg-muted rounded transition-colors disabled:opacity-30"
                        >
                            <ChevronLeft size={14} />
                        </button>
                        <span className="text-[10px] font-mono text-muted-foreground px-1">
                            {activeDoc + 1}/{docsContent.length}
                        </span>
                        <button
                            onClick={goNext}
                            disabled={activeDoc === docsContent.length - 1}
                            className="p-1.5 hover:bg-muted rounded transition-colors disabled:opacity-30"
                        >
                            <ChevronRight size={14} />
                        </button>
                    </div>
                </div>

                {/* TOC dropdown */}
                {showToc && (
                    <div className="border-b border-border bg-muted/50 px-4 py-3 animate-slide-up">
                        <p className="text-[9px] font-mono text-muted-foreground uppercase tracking-widest mb-2">
                            Table of Contents
                        </p>
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-0.5 max-h-60 overflow-y-auto">
                            {headings.map((h, i) => (
                                <button
                                    key={i}
                                    onClick={() => scrollToHeading(h.id)}
                                    className={`text-left text-[11px] font-mono py-1 rounded hover:bg-muted transition-colors truncate ${
                                        h.level === 1
                                            ? "font-bold text-foreground"
                                            : h.level === 2
                                              ? "pl-4 text-foreground"
                                              : "pl-8 text-muted-foreground"
                                    }`}
                                >
                                    {h.text}
                                </button>
                            ))}
                        </div>
                    </div>
                )}

                {/* Markdown content */}
                <div ref={contentRef} className="flex-1 overflow-y-auto">
                    <article className="max-w-4xl mx-auto px-6 py-8 docs-markdown animate-fade-in" key={activeDoc}>
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex, rehypeHighlight]}
                            components={{
                                h1: ({ children, ...props }) => {
                                    const id = String(children)
                                        .toLowerCase()
                                        .replace(/[^a-z0-9\s-]/g, "")
                                        .replace(/\s+/g, "-")
                                        .slice(0, 60);
                                    return (
                                        <h1 id={id} className="text-xl font-bold font-mono tracking-tight mt-0 mb-6 pb-3 border-b border-border" {...props}>
                                            {children}
                                        </h1>
                                    );
                                },
                                h2: ({ children, ...props }) => {
                                    const id = String(children)
                                        .toLowerCase()
                                        .replace(/[^a-z0-9\s-]/g, "")
                                        .replace(/\s+/g, "-")
                                        .slice(0, 60);
                                    return (
                                        <h2 id={id} className="text-base font-bold font-mono mt-10 mb-4 flex items-center gap-2" {...props}>
                                            <span className="text-accent">##</span>
                                            {children}
                                        </h2>
                                    );
                                },
                                h3: ({ children, ...props }) => {
                                    const id = String(children)
                                        .toLowerCase()
                                        .replace(/[^a-z0-9\s-]/g, "")
                                        .replace(/\s+/g, "-")
                                        .slice(0, 60);
                                    return (
                                        <h3 id={id} className="text-sm font-bold font-mono mt-8 mb-3 text-foreground" {...props}>
                                            {children}
                                        </h3>
                                    );
                                },
                                p: ({ children, ...props }) => (
                                    <p className="text-[13px] leading-relaxed text-foreground/85 mb-4 font-sans" {...props}>
                                        {children}
                                    </p>
                                ),
                                ul: ({ children, ...props }) => (
                                    <ul className="text-[13px] leading-relaxed text-foreground/85 mb-4 ml-4 space-y-1.5 list-none" {...props}>
                                        {children}
                                    </ul>
                                ),
                                ol: ({ children, ...props }) => (
                                    <ol className="text-[13px] leading-relaxed text-foreground/85 mb-4 ml-4 space-y-1.5 list-decimal" {...props}>
                                        {children}
                                    </ol>
                                ),
                                li: ({ children, ...props }) => (
                                    <li className="text-[13px] leading-relaxed pl-1" {...props}>
                                        <span className="text-accent mr-1.5 font-mono">&#x2022;</span>
                                        {children}
                                    </li>
                                ),
                                a: ({ href, children, ...props }) => (
                                    <a
                                        href={href}
                                        className="text-accent hover:underline font-mono text-[12px]"
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        {...props}
                                    >
                                        {children}
                                    </a>
                                ),
                                strong: ({ children, ...props }) => (
                                    <strong className="font-semibold text-foreground" {...props}>
                                        {children}
                                    </strong>
                                ),
                                blockquote: ({ children, ...props }) => (
                                    <blockquote
                                        className="border-l-2 border-accent pl-4 my-4 text-[12px] text-muted-foreground italic font-sans"
                                        {...props}
                                    >
                                        {children}
                                    </blockquote>
                                ),
                                hr: () => <hr className="border-border my-8" />,
                                code: ({ className, children, ...props }) => {
                                    const isInline = !className;
                                    if (isInline) {
                                        return (
                                            <code
                                                className="bg-muted border border-border px-1.5 py-0.5 rounded text-[11px] font-mono text-foreground"
                                                {...props}
                                            >
                                                {children}
                                            </code>
                                        );
                                    }
                                    return (
                                        <code className={`${className} text-[11px]`} {...props}>
                                            {children}
                                        </code>
                                    );
                                },
                                pre: ({ children, ...props }) => (
                                    <pre
                                        className="bg-[#fafafa] border border-border rounded-md p-4 my-4 overflow-x-auto text-[11px] font-mono leading-relaxed"
                                        {...props}
                                    >
                                        {children}
                                    </pre>
                                ),
                                table: ({ children, ...props }) => (
                                    <div className="my-6 overflow-x-auto border border-border rounded-md">
                                        <table className="w-full text-[11px] font-mono" {...props}>
                                            {children}
                                        </table>
                                    </div>
                                ),
                                thead: ({ children, ...props }) => (
                                    <thead className="bg-muted border-b border-border" {...props}>
                                        {children}
                                    </thead>
                                ),
                                th: ({ children, ...props }) => (
                                    <th
                                        className="text-left px-3 py-2 text-[10px] font-bold uppercase tracking-wider text-muted-foreground"
                                        {...props}
                                    >
                                        {children}
                                    </th>
                                ),
                                td: ({ children, ...props }) => (
                                    <td className="px-3 py-2 border-t border-border text-foreground/85" {...props}>
                                        {children}
                                    </td>
                                ),
                                tr: ({ children, ...props }) => (
                                    <tr className="hover:bg-muted/50 transition-colors" {...props}>
                                        {children}
                                    </tr>
                                ),
                            }}
                        >
                            {doc.content}
                        </ReactMarkdown>

                        {/* Navigation footer */}
                        <div className="mt-12 pt-6 border-t border-border flex items-center justify-between">
                            {activeDoc > 0 ? (
                                <button
                                    onClick={goPrev}
                                    className="group flex items-center gap-2 px-4 py-2.5 t-card hover:bg-muted transition-colors"
                                >
                                    <ChevronLeft size={14} className="text-muted-foreground group-hover:text-foreground transition-colors" />
                                    <div className="text-left">
                                        <p className="text-[9px] font-mono text-muted-foreground uppercase tracking-widest">Previous</p>
                                        <p className="text-[11px] font-mono font-bold truncate max-w-[200px]">
                                            {docsContent[activeDoc - 1].title.replace(/^\d+\s*—\s*/, "")}
                                        </p>
                                    </div>
                                </button>
                            ) : (
                                <div />
                            )}
                            {activeDoc < docsContent.length - 1 ? (
                                <button
                                    onClick={goNext}
                                    className="group flex items-center gap-2 px-4 py-2.5 t-card hover:bg-muted transition-colors"
                                >
                                    <div className="text-right">
                                        <p className="text-[9px] font-mono text-muted-foreground uppercase tracking-widest">Next</p>
                                        <p className="text-[11px] font-mono font-bold truncate max-w-[200px]">
                                            {docsContent[activeDoc + 1].title.replace(/^\d+\s*—\s*/, "")}
                                        </p>
                                    </div>
                                    <ChevronRight size={14} className="text-muted-foreground group-hover:text-foreground transition-colors" />
                                </button>
                            ) : (
                                <div />
                            )}
                        </div>
                    </article>
                </div>

                {/* Scroll to top */}
                {showScrollTop && (
                    <button
                        onClick={() => contentRef.current?.scrollTo({ top: 0, behavior: "smooth" })}
                        className="fixed bottom-6 right-6 p-2 bg-foreground text-white rounded-full shadow-lg hover:bg-foreground/80 transition-all animate-fade-in z-50"
                    >
                        <ArrowUp size={16} />
                    </button>
                )}
            </div>
        </div>
    );
}
