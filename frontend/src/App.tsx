import React, { useState, useEffect, useRef } from 'react';
import chlaLogo from './assets/image_copy.png';

// Define the structure for a message
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [documentCount, setDocumentCount] = useState<number | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const [activeQuery, setActiveQuery] = useState<{ query: string; history: any[] } | null>(null); 

  // Scroll to the latest message
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Initial welcome message and fetch document count
  useEffect(() => {
    setMessages([
      {
        id: Date.now(),
        text: "Hello! I'm here to help answer your medical questions and provide health information. Please remember that I'm not a substitute for professional medical advice, diagnosis, or treatment. How can I assist you today?",
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);

    // Fetch document count from backend
    const fetchDocumentCount = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/document_count');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setDocumentCount(data.count);
      } catch (error) {
        console.error('Error fetching document count:', error);
        setDocumentCount(0);
      }
    };

    fetchDocumentCount();
  }, []);

    useEffect(() => {
    if (!activeQuery) {
      return;
    }
    const { query, history } = activeQuery;
    const historyParam = encodeURIComponent(JSON.stringify(history));
    const queryParam = encodeURIComponent(query);
    const eventSourceUrl = `http://127.0.0.1:5000/chat?query=${queryParam}&history=${historyParam}`;
    const eventSource = new EventSource(eventSourceUrl);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.text) {
        setMessages((prevMessages) => {
          const newMessages = [...prevMessages];
          const lastMessage = newMessages[newMessages.length - 1]
          const updatedMessage = {
            ...lastMessage,
            text: lastMessage.text + data.text,
          };
          newMessages[newMessages.length - 1] = updatedMessage;
          return newMessages;
        });
      } else if (data.sources) {
        setMessages((prevMessages) => {
          const newMessages = [...prevMessages];
            const lastMessage = newMessages[newMessages.length - 1];
            const sourcesText = "\n\n**Sources:**\n" + data.sources.map((s: string) => `- ${s.replace(/_/g, " ").replace(".pdf", "")}`).join("\n");
            const updatedMessage = {
                ...lastMessage,
                text: lastMessage.text + sourcesText,
                timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            };
            newMessages[newMessages.length - 1] = updatedMessage;
            return newMessages;
        });
        setIsLoading(false);
        eventSource.close();
      } else if (data.error) {
         setMessages((prevMessages) => {
          const newMessages = [...prevMessages];
          newMessages[newMessages.length - 1].text = data.error;
          newMessages[newMessages.length - 1].timestamp = new Date().toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'});
          return newMessages;
        });
        setIsLoading(false);
        eventSource.close();
      }
    };
    eventSource.onerror = (err) => {
      console.error('EventSource failed:', err);
       setMessages((prevMessages) => {
          const newMessages = [...prevMessages];
          const lastMessage = newMessages[newMessages.length - 1];
          if (lastMessage && lastMessage.text === '') {
              lastMessage.text = "An error occurred while fetching the response. Please check the server logs.";
              lastMessage.timestamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
          }
          return newMessages;
      });
      setIsLoading(false);
      eventSource.close();
    };

    return () => {
      eventSource.close();
      setActiveQuery(null);
    };

  }, [activeQuery]);


  const sendMessage = async () => {
    if (input.trim() === '') return;

    const newUserMessage: Message = {
      id: Date.now(),
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    const historyForBackend = messages.map(msg => ({
      text: msg.text,
      isUser: msg.sender == "user",
    }));
    setMessages((prevMessages) => [
      ...prevMessages,
      newUserMessage,
      {id: Date.now() + 1, text: '', sender: 'assistant', timestamp: ''},
    ]);
    setActiveQuery({query: input, history: historyForBackend});
    setInput('');
    setIsLoading(true);
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !isLoading) {
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-purple-50 antialiased">
      {/* Centered Container for the entire chat interface */}
      <div className="relative flex flex-col flex-1 max-w-4xl w-full mx-auto my-4 rounded-3xl shadow-2xl overflow-hidden bg-white">
        {/* Header - Top Bar */}
        <header className="flex items-center p-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg rounded-t-3xl">
          <img src={chlaLogo} alt="CHLA Logo" className="h-16 w-16 mr-4 rounded-full object-contain bg-white p-2 shadow-inner" />
          <div>
            <h1 className="text-2xl font-bold">CHLA Health Librarian</h1> {/* Changed title here */}
            <p className="text-base font-light opacity-90">Children's Hospital Los Angeles</p>
          </div>
          {/* Removed star and heart icons from here */}
        </header>

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4 bg-gray-50">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[75%] p-4 rounded-xl shadow-md ${
                  message.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-white text-gray-800 rounded-bl-none border border-gray-200'
                }`}
              >
                <p className="text-base leading-relaxed">{message.text}</p>
                <span
                  className={`text-xs mt-2 block ${
                    message.sender === 'user' ? 'text-blue-200' : 'text-gray-500'
                  } text-right`}
                >
                  {message.timestamp}
                </span>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="max-w-[75%] p-4 rounded-xl shadow-md bg-white rounded-bl-none border border-gray-200">
                <div className="flex items-center space-x-2">
                  <div className="dot-flashing"></div>
                  <div className="dot-flashing dot-flashing-2"></div>
                  <div className="dot-flashing dot-flashing-3"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="p-6 bg-white border-t border-gray-100 shadow-inner">
          <div className="flex items-center space-x-4">
            <input
              type="text"
              className="flex-1 p-4 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm text-gray-700 placeholder-gray-400 text-base"
              placeholder="Ask me about health topics, symptoms, or medical questions..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              className={`p-4 rounded-full bg-blue-600 text-white flex items-center justify-center shadow-lg transition-all duration-200 ${
                isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700 active:scale-95'
              }`}
              disabled={isLoading}
            >
              {/* Simple SVG for send icon */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth={1.5}
                stroke="currentColor"
                className="w-6 h-6"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"
                />
              </svg>
            </button>
          </div>
        </div>

        {/* Footer Disclaimer */}
        <footer className="w-full bg-gray-100 p-4 text-center text-xs text-gray-600 border-t border-gray-200 rounded-b-3xl">
          <p>
            This AI assistant provides general health information and is not a substitute for
            professional medical advice, diagnosis, or treatment. Always seek the advice of your
            physician or other qualified health provider with any questions you may have.
            {documentCount !== null && (
              <span className="block mt-1 text-gray-500">
                Data from {documentCount} documents.
              </span>
            )}
          </p>
          {/* New contact information */}
          <p className="mt-2">
            If you have questions about our library, contact us here:
            <span className="block font-semibold text-gray-700">Health Library</span>
            <span className="block font-semibold text-gray-700">323-361-2254</span>
          </p>
        </footer>
      </div> {/* End of main centered container */}

      {/* Simple CSS for loading dots (can be moved to index.css if preferred) */}
      <style>{`
        .dot-flashing {
          position: relative;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background-color: #9ca3af; /* Tailwind gray-400 */
          color: #9ca3af;
          animation: dotFlashing 1s infinite linear alternate;
          animation-delay: 0s;
        }

        .dot-flashing-2 {
          animation-delay: 0.2s;
        }

        .dot-flashing-3 {
          animation-delay: 0.4s;
        }

        @keyframes dotFlashing {
          0% {
            background-color: #9ca3af;
          }
          50%,
          100% {
            background-color: #e5e7eb; /* Tailwind gray-200 */
          }
        }
      `}</style>
    </div>
  );
}
