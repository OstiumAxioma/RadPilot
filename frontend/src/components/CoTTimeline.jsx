import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Brain, Zap, Activity, Clock, AlertTriangle, CheckCircle2, Loader2 } from 'lucide-react';

/**
 * 医疗级 ReAct 思维链与 CoT 步骤流实时可视化组件
 * 支持流式动态渲染、质检验收告警与实时执行状态展示
 */
export default function CoTTimeline({ steps = [], totalElapsed = 0, isStreaming = false, activeStatus = '' }) {
    const [isExpanded, setIsExpanded] = useState(true);

    if ((!steps || steps.length === 0) && !isStreaming) return null;

    return (
        <div style={{
            margin: '8px 0',
            border: '1px solid #e2e8f0',
            borderRadius: '6px',
            background: '#ffffff',
            overflow: 'hidden',
            boxShadow: '0 1px 3px rgba(0,0,0,0.02)'
        }}>
            {/* 顶栏: 摘要与折叠按钮 */}
            <div
                onClick={() => setIsExpanded(!isExpanded)}
                style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '6px 10px',
                    background: '#f8fafc',
                    borderBottom: isExpanded ? '1px solid #e2e8f0' : 'none',
                    cursor: 'pointer',
                    userSelect: 'none'
                }}
            >
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, fontWeight: 700, color: '#0f172a' }}>
                    {isStreaming ? (
                        <Loader2 size={13} className="animate-spin" style={{ color: '#0284c7' }} />
                    ) : (
                        <Brain size={13} style={{ color: '#0284c7' }} />
                    )}
                    <span>ReAct 多步自主精修思维链</span>
                    <span style={{
                        background: isStreaming ? '#fef3c7' : '#e0f2fe',
                        color: isStreaming ? '#b45309' : '#0369a1',
                        padding: '1px 6px',
                        borderRadius: 10,
                        fontSize: 10,
                        fontWeight: 600
                    }}>
                        {isStreaming ? `第 ${steps.length + 1} 步执行中...` : `${steps.length} 步闭环迭代`}
                    </span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 10.5, color: '#64748b' }}>
                    {totalElapsed > 0 && (
                        <span style={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                            <Clock size={11} /> {totalElapsed}ms
                        </span>
                    )}
                    {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
                </div>
            </div>

            {/* 展开内容: 思维链时间线 */}
            {isExpanded && (
                <div style={{ padding: '8px 10px', display: 'flex', flexDirection: 'column', gap: 8 }}>
                    {steps.map((step, idx) => (
                        <div
                            key={idx}
                            style={{
                                display: 'flex',
                                gap: 8,
                                position: 'relative',
                                paddingLeft: 4
                            }}
                        >
                            {/* 左侧节点指示器 */}
                            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                <div style={{
                                    width: 18,
                                    height: 18,
                                    borderRadius: '50%',
                                    background: '#0284c7',
                                    color: '#ffffff',
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    fontSize: 9.5,
                                    fontWeight: 700
                                }}>
                                    {step.step_index || idx + 1}
                                </div>
                                {(idx < steps.length - 1 || isStreaming) && (
                                    <div style={{ width: 2, flex: 1, background: '#e2e8f0', margin: '3px 0' }} />
                                )}
                            </div>

                            {/* 右侧详细步骤卡片 */}
                            <div style={{
                                flex: 1,
                                background: '#f8fafc',
                                border: '1px solid #f1f5f9',
                                borderRadius: 5,
                                padding: '6px 8px',
                                fontSize: 11
                            }}>
                                {/* 1. Thought 思考 */}
                                {step.thought && (
                                    <div style={{ marginBottom: 4 }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 4, color: '#475569', fontWeight: 600, fontSize: 10.5, marginBottom: 2 }}>
                                            <span>🧠 诊断与规划 (Thought):</span>
                                        </div>
                                        <div style={{ color: '#1e293b', lineHeight: 1.5, paddingLeft: 4 }}>
                                            {step.thought}
                                        </div>
                                    </div>
                                )}

                                {/* 2. Action 执行动作 */}
                                {step.action_name && (
                                    <div style={{
                                        display: 'flex',
                                        alignItems: 'center',
                                        flexWrap: 'wrap',
                                        gap: 4,
                                        margin: '4px 0',
                                        padding: '3px 6px',
                                        background: '#ffffff',
                                        borderRadius: 4,
                                        border: '1px solid #e2e8f0'
                                    }}>
                                        <Zap size={11} style={{ color: '#d97706' }} />
                                        <span style={{ fontWeight: 700, color: '#0f172a', fontFamily: 'var(--font-mono)' }}>
                                            {step.action_name}
                                        </span>
                                        {step.action_params && Object.keys(step.action_params).length > 0 && (
                                            <span style={{ color: '#64748b', fontSize: 10, fontFamily: 'var(--font-mono)' }}>
                                                ({JSON.stringify(step.action_params)})
                                            </span>
                                        )}
                                    </div>
                                )}

                                {/* 3. Observation 物理观察 */}
                                {step.observation && (
                                    <div style={{
                                        marginTop: 4,
                                        display: 'flex',
                                        alignItems: 'flex-start',
                                        gap: 4,
                                        color: '#0369a1',
                                        background: '#f0f9ff',
                                        padding: '4px 6px',
                                        borderRadius: 4,
                                        fontSize: 10.5,
                                        lineHeight: 1.4
                                    }}>
                                        <Activity size={12} style={{ marginTop: 1, flexShrink: 0 }} />
                                        <span>{step.observation}</span>
                                    </div>
                                )}

                                {/* 4. 质检验收告警 (Verification Gate) */}
                                {step.warnings && step.warnings.length > 0 && (
                                    <div style={{
                                        marginTop: 4,
                                        display: 'flex',
                                        flexDirection: 'column',
                                        gap: 2,
                                        color: '#b45309',
                                        background: '#fffbeb',
                                        border: '1px solid #fef3c7',
                                        padding: '4px 6px',
                                        borderRadius: 4,
                                        fontSize: 10.5,
                                        lineHeight: 1.4
                                    }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontWeight: 700 }}>
                                            <AlertTriangle size={12} />
                                            <span>质检验收门控触发告警:</span>
                                        </div>
                                        {step.warnings.map((w, wIdx) => (
                                            <div key={wIdx} style={{ paddingLeft: 16 }}>
                                                {w}
                                            </div>
                                        ))}
                                    </div>
                                )}
                            </div>
                        </div>
                    ))}

                    {/* 正在进行的实时流式动态占位 */}
                    {isStreaming && (
                        <div style={{ display: 'flex', gap: 8, paddingLeft: 4 }}>
                            <div style={{
                                width: 18,
                                height: 18,
                                borderRadius: '50%',
                                background: '#f59e0b',
                                color: '#ffffff',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                                fontSize: 9.5,
                                fontWeight: 700
                            }}>
                                {steps.length + 1}
                            </div>
                            <div style={{
                                flex: 1,
                                background: '#fffbeb',
                                border: '1px dashed #fde68a',
                                borderRadius: 5,
                                padding: '6px 8px',
                                fontSize: 11,
                                color: '#92400e',
                                display: 'flex',
                                alignItems: 'center',
                                gap: 6
                            }}>
                                <Loader2 size={12} className="animate-spin" />
                                <span>{activeStatus || 'Gemini 多模态视觉推理与解剖质检中...'}</span>
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
