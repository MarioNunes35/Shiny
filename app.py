# Adicionar esses componentes ao app

def create_quick_actions():
    """Cria botões de ação rápida para perguntas comuns"""
    return ui.div(
        {"class": "quick-actions"},
        ui.h4("Perguntas Frequentes", style="margin-bottom: 12px;"),
        ui.div(
            {"class": "quick-action-grid"},
            ui.input_action_button(
                "quick_plot", 
                "📊 Como criar gráficos?",
                class_="quick-btn"
            ),
            ui.input_action_button(
                "quick_import", 
                "📥 Importar dados",
                class_="quick-btn"
            ),
            ui.input_action_button(
                "quick_analysis", 
                "📈 Análise estatística",
                class_="quick-btn"
            ),
            ui.input_action_button(
                "quick_fitting", 
                "📉 Curve fitting",
                class_="quick-btn"
            )
        )
    )

def create_status_indicators():
    """Indicadores de status do sistema"""
    return ui.div(
        {"class": "status-bar"},
        ui.span(
            {"class": "status-item"},
            "🟢 Claude: " + ("Ativo" if client else "Inativo"),
            style="color: " + ("#98c379" if client else "#e06c75")
        ),
        ui.span(
            {"class": "status-item"},
            f"📚 RAG: {'Ativo' if HAVE_RAG_DEPS else 'Inativo'}",
            style="color: " + ("#98c379" if HAVE_RAG_DEPS else "#e06c75")
        ),
        ui.span(
            {"class": "status-item"},
            f"👤 {current_user()}"
        )
    )

def create_export_options():
    """Opções para exportar conversa"""
    return ui.div(
        {"class": "export-options"},
        ui.input_action_button(
            "export_md",
            "📄 Exportar Markdown",
            class_="btn-small"
        ),
        ui.input_action_button(
            "export_pdf",
            "📑 Exportar PDF",
            class_="btn-small"
        ),
        ui.input_action_button(
            "copy_last",
            "📋 Copiar última resposta",
            class_="btn-small"
        )
    )

# CSS adicional para os novos componentes
ADDITIONAL_CSS = """
/* Quick Actions Grid */
.quick-actions {
    padding: 16px;
    background: #141414;
    border-radius: 8px;
    margin: 16px;
}

.quick-action-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 8px;
}

.quick-btn {
    padding: 12px 8px;
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 6px;
    color: #888;
    font-size: 12px;
    cursor: pointer;
    text-align: center;
    transition: all 0.2s;
    min-height: 44px;
}

.quick-btn:hover {
    background: #2a2a2a;
    color: #e0e0e0;
    border-color: #3a3a3a;
}

/* Status Bar */
.status-bar {
    display: flex;
    gap: 16px;
    padding: 8px 16px;
    background: #0e0e0e;
    border-top: 1px solid #2a2a2a;
    font-size: 11px;
    justify-content: space-between;
    flex-wrap: wrap;
}

.status-item {
    display: flex;
    align-items: center;
    gap: 4px;
}

/* Export Options */
.export-options {
    display: flex;
    gap: 8px;
    padding: 8px;
    justify-content: flex-end;
}

.btn-small {
    padding: 6px 10px;
    font-size: 11px;
    background: transparent;
    border: 1px solid #2a2a2a;
    color: #888;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
}

.btn-small:hover {
    background: #1a1a1a;
    color: #e0e0e0;
}

/* Dark mode toggle */
@media (prefers-color-scheme: light) {
    /* Suporte para light mode se necessário */
    .light-mode {
        background: #ffffff;
        color: #000000;
    }
}

/* Loading skeleton */
.skeleton {
    animation: skeleton-loading 1s linear infinite alternate;
}

@keyframes skeleton-loading {
    0% {
        background-color: #1a1a1a;
    }
    100% {
        background-color: #2a2a2a;
    }
}

/* Tooltips */
.tooltip {
    position: relative;
}

.tooltip::after {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 125%;
    left: 50%;
    transform: translateX(-50%);
    background: #2a2a2a;
    color: #e0e0e0;
    padding: 4px 8px;
    border-radius: 4px;
    font-size: 11px;
    white-space: nowrap;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.2s;
}

.tooltip:hover::after {
    opacity: 1;
}
"""

# Handler para quick actions
@reactive.Effect
@reactive.event(input.quick_plot)
def handle_quick_plot():
    """Envia pergunta sobre criação de gráficos"""
    question = "Como criar um gráfico de dispersão com linha de tendência no OriginPro?"
    ui.update_text_area("prompt", value=question)
    # Trigger send automaticamente
    
@reactive.Effect
@reactive.event(input.quick_import)
def handle_quick_import():
    """Envia pergunta sobre importação de dados"""
    question = "Como importar dados de um arquivo Excel para o OriginPro?"
    ui.update_text_area("prompt", value=question)
    
@reactive.Effect
@reactive.event(input.quick_analysis)
def handle_quick_analysis():
    """Envia pergunta sobre análise estatística"""
    question = "Como fazer análise de regressão linear no OriginPro?"
    ui.update_text_area("prompt", value=question)
    
@reactive.Effect
@reactive.event(input.quick_fitting)
def handle_quick_fitting():
    """Envia pergunta sobre curve fitting"""
    question = "Como ajustar uma curva exponencial aos meus dados no OriginPro?"
    ui.update_text_area("prompt", value=question)







