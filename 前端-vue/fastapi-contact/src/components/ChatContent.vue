<template>
    <div class="chat-container">
        <div class="model-selector-container">
            <select id="model-selector" v-model="curModel">
                <option v-for="(model, index) in models" :value="model" :key="index">{{ model }}</option>
            </select>
        </div>
        <div class="messages-container">
            <div v-for="(msg, index) in messages" class="single-message" :key="index">
                <!-- 角色图标 -->
                <div class="message-icon" :class="msg.role">
                    <svg v-if="msg.role === 'user'" viewBox="0 0 24 24" width="24" height="24">
                        <path fill="currentColor"
                            d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z" />
                    </svg>
                    <svg v-else viewBox="0 0 24 24" width="24" height="24">
                        <path fill="currentColor"
                            d="M13 14.9c1.5-.4 3.1-.9 4.7-1.4.5-.2.8-.8.6-1.3-.2-.5-.8-.8-1.3-.6-1.6.5-3.1 1-4.7 1.4-.6.2-.9.8-.7 1.4.2.4.6.6 1 .6zm-3.6-3.6c.4.4 1 .4 1.4 0 .4-.4.4-1 0-1.4-.4-.4-1-.4-1.4 0-.4.4-.4 1 0 1.4zm5.5 5.5c.4.4 1 .4 1.4 0 .4-.4.4-1 0-1.4-.4-.4-1-.4-1.4 0-.4.4-.4 1 0 1.4zM12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm0 18c-4.4 0-8-3.6-8-8s3.6-8 8-8 8 3.6 8 8-3.6 8-8 8z" />
                    </svg>
                </div>

                <!-- 消息内容区域 -->
                <div :class="['message-content', msg.role]">
                    <!-- 角色名称 -->
                    <div class="message-role-name">
                        {{ getRoleName(msg.role) }}
                    </div>
                    <!-- 消息内容 -->
                    <div :class="['message', msg.role]" v-html="renderMarkdown(msg.content)"></div>
                </div>
            </div>
            <div v-if="waiting" class="message assistant">
                {{ content }}
            </div>
            <p>请开始你的对话~</p>
        </div>
        <div class="input-area">
            <input v-model="inputMessage" @keyup.enter="sendMessage" placeholder="输入消息..." />
            <button @click="sendMessage">发送</button>
        </div>
    </div>
</template>


<script>
import { marked } from 'marked';
import DOMPurify from 'dompurify';

export default {
    data() {
        return {
            inputMessage: '',
            messages: [],
            content: "加载中...",
            waiting: false,
            curModel: 'Deepseek',
            models: ['Deepseek', 'Chatgpt', '通义千问']
        };
    },
    methods: {
        getRoleName(role) {
            const roleNames = {
                'user': '用户',
                'assistant': this.curModel,
                'system': '系统'
            };
            return roleNames[role] || '未知角色';
        },
        renderMarkdown(wcontent) {
            const rawHtml = marked(wcontent || '');
            return DOMPurify.sanitize(rawHtml);
        },
        async sendMessage() {
            if (!this.inputMessage.trim()) return;

            this.messages.push({ "role": 'user', "model": this.curModel, "content": this.inputMessage });
            this.inputMessage = '';
            this.waiting = true;

            try {
                const response = await fetch('http://127.0.0.1:8000/message', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ "message": this.messages })
                });

                const reader = response.body.getReader();
                const decoder = new TextDecoder('utf-8');
                let tmp = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) {
                        break;
                    }
                    const chunk = decoder.decode(value, { stream: true }).replace(/^data: /, "");
                    console.log('Received chunk:', chunk);
                    tmp += chunk;
                }

                this.messages.push({
                    "role": 'assistant',
                    "model": this.curModel,
                    "content": tmp
                });

            } catch (error) {
                console.error('请求失败:', error);
                this.messages.push({
                    role: 'assistant',
                    "model": this.curModel,
                    content: '请求API时出错，请检查控制台' + error.message
                });
            } finally {
                this.waiting = false;
            }
        }
    }
};
</script>

<style>
#app {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;

    width: 100%;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    height: 800vh;
    display: flex;
    justify-content: center;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    padding: 20px;
    box-sizing: border-box;
}

/* 固定宽度的聊天容器 */
.chat-container {
    width: 100%;
    max-width: 800px;
    /* 最大宽度限制 */
    height: 90vh;
    /* 固定高度 */
    background-color: white;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    display: flex;
    flex-direction: column;
    position: relative;
    /* 为固定输入区域做准备 */
    overflow: hidden;
}

/* 消息容器 - 可滚动区域 */
.messages-container {
    flex: 1;
    overflow-y: auto;
    padding: 15px 20px;
    display: flex;
    flex-direction: column;
    gap: 15px;
}

.single-message {
    display: flex;
    gap: 12px;
}

/* 用户消息反转图标和内容位置 */
.single-message.user {
    flex-direction: row-reverse;
}

.message-icon {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: 16px;
}

.message-icon.user {
    background-color: #6a11cb;
    color: white;
    align-items: right;
}

.message-icon.assistant {
    background-color: #e2e8f0;
    color: #2d3748;
}

.message-content {
    max-width: 85%;
    display: flex;
    flex-direction: column;
}

.message-role-name {
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 6px;
    padding: 0 8px;
    color: #1e293b;
}

/* 用户名称右对齐 */
.single-message.user .message-role-name {
    text-align: right;
    color: #6a11cb;
}

/* 助手名称左对齐 */
.single-message.assistant .message-role-name {
    text-align: left;
    color: #4a5568;
}

/* 保留原有的消息样式 */
.message {
    padding: 12px 16px;
    border-radius: 12px;
    line-height: 1.5;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 用户消息样式 - 保留渐变背景 */
.message.user {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    border-bottom-right-radius: 4px;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    align-items: right;
}


/* 助手消息样式 */
.message.assistant {
    background-color: #f7fafc;
    color: #2d3748;
    border: 1px solid #e2e8f0;
    border-bottom-left-radius: 4px;
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
}

/* 等待消息样式 */
.message.assistant.waiting {
    padding: 16px;
    background-color: #edf2f7;
    border-radius: 12px;
}

/* 初始提示文字 */
.messages-container>p {
    text-align: center;
    color: #a0aec0;
    padding: 20px;
    font-style: italic;
}

/* 固定在底部的输入区域 */
.input-area {
    position: absolute;
    /* 绝对定位固定在底部 */
    bottom: 0;
    left: 0;
    right: 0;
    display: flex;
    padding: 20px;
    background-color: white;
    border-top: 1px solid #edf2f7;
    z-index: 10;
    /* 确保在消息上方 */
}

input {
    flex: 1;
    padding: 15px 20px;
    border: 1px solid #e2e8f0;
    border-radius: 30px;
    font-size: 16px;
    outline: none;
    transition: all 0.3s;
    background-color: white;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.05);
}

input:focus {
    border-color: #4299e1;
    box-shadow: 0 0 0 3px rgba(66, 153, 225, 0.2);
}

button {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: white;
    border: none;
    border-radius: 30px;
    padding: 15px 30px;
    margin-left: 15px;
    cursor: pointer;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s;
    box-shadow: 0 4px 10px rgba(37, 117, 252, 0.3);
}

button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(37, 117, 252, 0.4);
}

button:active {
    transform: translateY(0);
}



@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 响应式设计 */
@media (max-width: 600px) {
    .chat-container {
        height: 95vh;
        border-radius: 15px;
    }

    .messages-container {
        padding: 15px;
        padding-bottom: 70px;
        /* 移动端调整底部内边距 */
    }

    .message {
        max-width: 85%;
        padding: 12px 16px;
    }

    .input-container {
        padding: 15px;
    }

    input,
    button {
        padding: 12px 20px;
    }

    button {
        padding: 12px 25px;
    }

    #model-selector {
        background-color: rgba(30, 41, 59, 0.9);
        border-color: #475569;
        color: #f1f5f9;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%23cbd5e1' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");
    }
}


.model-selector-container {
    position: relative;
    max-width: 300px;
    margin: 1rem auto;
}

#model-selector {
    /* 基础样式 */
    width: 100%;
    padding: 12px 48px 12px 16px;
    font-size: 1rem;
    font-family: 'Segoe UI', system-ui, sans-serif;

    /* 边框与背景 */
    background-color: rgba(255, 255, 255, 0.9);
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);

    /* 自定义下拉箭头 */
    appearance: none;
    -webkit-appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' fill='%236b7280' viewBox='0 0 16 16'%3E%3Cpath d='M8 11L3 6h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 16px center;
    background-size: 16px;

    /* 过渡效果 */
    transition: all 0.3s ease;
    cursor: pointer;
}

/* 悬停效果 */
#model-selector:hover {
    border-color: #94a3b8;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

/* 聚焦效果 */
#model-selector:focus {
    outline: none;
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}


/* 选项样式 */
#model-selector option {
    padding: 10px;
    background: #fff;
    color: #1e293b;
}

/* 禁用状态 */
#model-selector:disabled {
    opacity: 0.7;
    cursor: not-allowed;
}
</style>