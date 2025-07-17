import React, { useState, useRef, useEffect} from 'react';
import { Send, User, BookOpen, AlertTriangle } from 'lucide-react';
import { Remarkable } from 'remarkable';

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  sources?: string[];
  error?: boolean;
}

interface ChatHistoryItem {
  text: string;
  isUser: boolean;
}

const ButterflyIcon = ({ className = "w-6 h-6" } : { className?: string}) => (
  <svg className={className} viewBox="0 0 200 150" xmlns="https://www.w3.org/2000/svg">
    <path d="M100 75C100 75 80 50 40 40C0 30 0 75 0 75C0 75 0 120 40 110C80 100 100 75 100 75Z" fill="#00AEEF"/>
    <path d="M100 75C100 75 120 50 160 40C200 30 200 75 200 75C200 75 200 120 160 110C120 100 100 75 100 75Z" fill="#FDB913"/>
    <path d="M100 75C100 75 80 100 40 110C0 120 0 75 0 75C0 75 0 30 40 40C80 50 100 75 100 75Z" fill="#8DC63F"/>
    <path d="M100 75C100 75 120 100 160 110C200 120 200 75 200 75C200 75 200 30 160 40C120 50 100 75 100 75Z" fill="#E94E1B"/>
  </svg>
);

const ChatHeader = () => (
  <header className="bg-white/80 backdrop-blur-lg border-b border-gray-200 sticky top-0 z-10">
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
      <div className="flex items-center justify-between h-16">
        <div className="flex items-center space-x-3">
          <ButterflyIcon className="w-9 h-9" />
          <div>
            <h1 className="text-lg font-bold text-gray-800">CHLA Health Librarian</h1>
            <p className="text-xs text-gray-500">Your guide to our health library</p>
          </div>
        </div>
      </div>
    </div>
  </header>
);

const MessageSources = ({ sources }: { sources:string[] }) => (
  <div className="mt-3 pt-3 border-t border-blue-200/50">
    <h4 className="text-xs font-semibold text-gray-600 mb-2 flex items-center">
      <BookOpen className="w-4 h-4 mr-2" />
      Sources
    </h4>
    <div className="flex flex-wrap gap-2">
      {sources.map((s, i) => (
        <span key={i} className="px-2.5 py-1 text-xs font-medium text-blue-800 bg-blue-100 rounded-full">
          {s.replace(/_/g, " ").replace(/\.pdf$/i, "")}
        </span>
      ))}
    </div>
  </div>
);

const MessageBubble = ({ msg }: { msg: Message }) => {
  const md = new Remarkable();
  const renderedText = md.render(msg.text);

  return (
    <div className={`flex items-start gap-3 ${msg.isUser ? 'flex-row-reverse' : ''}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${msg.isUser ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}>
        {msg.isUser ? <User className="w-4 h-4" /> : <ButterflyIcon className="w-5 h-5" />}
      </div>
      <div className={`w-full max-w-xl p-4 rounded-2xl ${msg.isUser ? 'bg-blue-500 text-white rounded-br-none' : 'bg-white text-gray-800 border border-gray-200 rounded-bl-none'}`}>
        {msg.error ? (
          <div className="flex items-center gap-3 text-red-700 bg-red-50 p-3 rounded-lg">
            <AlertTriangle className="w-5 h-5 flex-shrink-0" />
            <p className="text-sm font-medium">{msg.text}</p>
          </div>
        ) : (
          <div className="prose prose-sm max-w-none" dangerouslySetInnerHTML={{ __html: renderedText }} />
        )}
        {msg.sources && msg.sources.length > 0 && <MessageSources sources={msg.sources} />}
      </div>
    </div>
  );
};

const TypingIndicator = () => (
  <div className="flex items-start gap-3">
    <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-gray-200">
      <ButterflyIcon className="w-5 h-5" />
    </div>
    <div className="w-full max-w-xl p-4 rounded-2xl bg-white border border-gray-200 rounded-bl-none">
      <div className="flex items-center space-x-1.5">
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></span>
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></span>
        <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
      </div>
    </div>
  </div>
);


// ----- MAIN CHAT COMPONENT -----
const CHLAMedicalAssistant: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I am CHLA's Automated Health Librarian. I can help answer questions based on our health education library. How can I assist you today?",
      isUser: false,
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth"});
  }, [messages, isTyping])

  const textAreaRef = useRef<HTMLTextAreaElement>(null);
  useEffect(() => {
    if (textAreaRef.current) {
      textAreaRef.current.style.height = 'auto',
      textAreaRef.current.style.height = `${textAreaRef.current.scrollHeight}px`;
    }
  }, [inputText]);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;
    const userMessage: Message = {
      id: Date.now(),
      text: inputText,
      isUser: true,
    };
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    const conversationHistoryForBackend: ChatHistoryItem[] = messages.slice(1).map(msg => ({ text: msg.text, isUser: msg.isUser }));
    conversationHistoryForBackend.push({ text: userMessage.text, isUser: true});

    let aiResponseText = '';
    let collectedSources: string[] = [];
    const aiMessageId = Date.now() + 1;

    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          query: userMessage.text,
          history: conversationHistoryForBackend.slice(0, -1)
        }),
      });
      if (!response.ok || !response.body) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
      }
      setIsTyping(false);
      setMessages(prev => [...prev, { id: aiMessageId, text: '', isUser: false }]);

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true){
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, {stream: true});
        const lines = chunk.split('\n\n').filter(line => line.trim() !== '');
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.substring(6);
            try {
              const parsedData = JSON.parse(data);
              if (parsedData.sources) {
                collectedSources = parsedData.sources;
              } else if (parsedData.error) {
                throw new Error(parsedData.error);
              }
            } catch (e) {
              aiResponseText += data;
              setMessages(prev => prev.map(msg => msg.id === aiMessageId ? { ...msg, text: aiResponseText } : msg));
            }
          }
        }
      }
      setMessages(prev => prev.map(msg => msg.id === aiMessageId ? {...msg, text: aiResponseText, sources: collectedSources } : msg));
    } catch (error: any) {
      setIsTyping(false);
      setMessages(prev => [...prev, {
        id: aiMessageId,
        text: `Sorry, an error occurred: ${error.message}. Please try again!`,
        isUser: false,
        error: true,
      }]);
    }
  };
  const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };
  return (
    <div className="flex flex-col h-screen bg-gray-50 font-sans">
      <ChatHeader />
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="space-y-6">
            {messages.map((msg) => <MessageBubble key={msg.id} msg={msg} />)}
            {isTyping && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </main>
      <footer className="bg-white border-t border-gray-200">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py3">
          <div className="flex items-center space-x-3">
            <textarea
              ref={textAreaRef}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about my wealth of knowledge on various health topics, symptoms, and medical procedures, curated by CHLA."
              className="flex-1 w-full px-4 py-2.5 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-gray-50"
              rows={1} />
            <button onClick={handleSendMessage} disabled={!inputText.trim() || isTyping} className="flex-shrink-0 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-colors duration-200 shadow-sm hover:shadow-md focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500" aria-label="Send message">
              <Send className="w-5 h-5" />
            </button>
          </div>
          <p className="text-xs text-gray-400 mt-2 text-center">
            This AI Health Librarian is for informational purposes only and is not a substitute for professional medical advice. If you do not find what you need using this service, you may call this number for CHLA's professional Health Library: +1 (323) 361-2254.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default CHLAMedicalAssistant;