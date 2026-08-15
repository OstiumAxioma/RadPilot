import React from 'react';

/**
 * 轻量级医疗级 Markdown 格式渲染器
 * 解析并优雅排版粗体 (**text**)、标题 (###)、列表 (- list / 1. list)、代码块与分割线
 */
export default function MarkdownRenderer({ content }) {
    if (!content) return null;

    const lines = content.split('\n');
    const elements = [];

    let currentList = [];
    let listType = null; // 'ul' | 'ol'

    const flushList = () => {
        if (currentList.length > 0) {
            if (listType === 'ol') {
                elements.push(
                    <ol key={`ol-${elements.length}`} style={{ margin: '4px 0 6px 18px', padding: 0, fontSize: 11.5, lineHeight: 1.6 }}>
                        {currentList.map((item, idx) => (
                            <li key={idx} style={{ marginBottom: 2 }}>{renderInline(item)}</li>
                        ))}
                    </ol>
                );
            } else {
                elements.push(
                    <ul key={`ul-${elements.length}`} style={{ margin: '4px 0 6px 16px', padding: 0, fontSize: 11.5, lineHeight: 1.6, listStyleType: 'disc' }}>
                        {currentList.map((item, idx) => (
                            <li key={idx} style={{ marginBottom: 2 }}>{renderInline(item)}</li>
                        ))}
                    </ul>
                );
            }
            currentList = [];
            listType = null;
        }
    };

    lines.forEach((line, index) => {
        const trimmed = line.trim();

        // 空行
        if (!trimmed) {
            flushList();
            return;
        }

        // 分割线 ---
        if (trimmed === '---' || trimmed === '***') {
            flushList();
            elements.push(<hr key={`hr-${index}`} style={{ border: 'none', borderTop: '1px solid #e2e8f0', margin: '8px 0' }} />);
            return;
        }

        // 标题 ###, ##, #
        if (trimmed.startsWith('### ')) {
            flushList();
            elements.push(
                <h4 key={`h4-${index}`} style={{ margin: '8px 0 4px', fontSize: 12, fontWeight: 700, color: '#0f172a' }}>
                    {renderInline(trimmed.slice(4))}
                </h4>
            );
            return;
        }
        if (trimmed.startsWith('## ')) {
            flushList();
            elements.push(
                <h3 key={`h3-${index}`} style={{ margin: '10px 0 4px', fontSize: 12.5, fontWeight: 700, color: '#0f172a' }}>
                    {renderInline(trimmed.slice(3))}
                </h3>
            );
            return;
        }
        if (trimmed.startsWith('# ')) {
            flushList();
            elements.push(
                <h2 key={`h2-${index}`} style={{ margin: '12px 0 6px', fontSize: 13, fontWeight: 800, color: '#0f172a' }}>
                    {renderInline(trimmed.slice(2))}
                </h2>
            );
            return;
        }

        // 无序列表 - 或 *
        if (trimmed.startsWith('- ') || trimmed.startsWith('* ')) {
            if (listType !== 'ul') flushList();
            listType = 'ul';
            currentList.push(trimmed.slice(2));
            return;
        }

        // 有序列表 1. 2.
        const olMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
        if (olMatch) {
            if (listType !== 'ol') flushList();
            listType = 'ol';
            currentList.push(olMatch[2]);
            return;
        }

        // 普通段落
        flushList();
        elements.push(
            <p key={`p-${index}`} style={{ margin: '0 0 6px', fontSize: 11.5, lineHeight: 1.6, color: '#1e293b' }}>
                {renderInline(trimmed)}
            </p>
        );
    });

    flushList();

    return <div className="markdown-body" style={{ wordBreak: 'break-word' }}>{elements}</div>;
}

/**
 * 递归解析行内加粗 (**bold**)、行内代码 (`code`)
 */
function renderInline(text) {
    if (!text) return null;

    // 匹配 **bold** 和 `code`
    const regex = /(\*\*.*?\*\*|`.*?`)/g;
    const parts = text.split(regex);

    return parts.map((part, idx) => {
        if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
            return (
                <strong key={idx} style={{ fontWeight: 700, color: '#0f172a' }}>
                    {part.slice(2, -2)}
                </strong>
            );
        }
        if (part.startsWith('`') && part.endsWith('`') && part.length >= 2) {
            return (
                <code
                    key={idx}
                    style={{
                        background: '#f1f5f9',
                        color: '#0284c7',
                        padding: '1px 4px',
                        borderRadius: 3,
                        fontSize: 10.5,
                        fontFamily: 'var(--font-mono)'
                    }}
                >
                    {part.slice(1, -1)}
                </code>
            );
        }
        return part;
    });
}
