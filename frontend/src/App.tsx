import React, { useState, useEffect, useRef } from 'react';
// Ensure this path is correct for your CHLA logo image.
// If your image is named differently or in a different folder, adjust './assets/image_copy.png' accordingly.
const chlaLogo = new URL('./assets/image_copy.png', import.meta.url).href;

// This tells the frontend to make requests to port 5000, fixing the broken PDF links.
const API_URL = ''; 

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

interface HistoryItem {
  isUser: boolean;
  content: string;
}

interface Source {
  filename: string;
  url: string;
}

const renderMarkdown = (markdownText: string) => {
  let htmlText = markdownText;
  // Bold text: **text**
  htmlText = htmlText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  // Links: [text](url)
  htmlText = htmlText.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline">$1</a>');
  // Unordered lists: - item
  // This regex looks for one or more lines starting with "- "
  htmlText = htmlText.replace(/(\n- (.*))+/g, (match) => {
    const items = match.trim().split('\n').map(item => `<li>${item.substring(2)}</li>`).join('');
    return `<ul>${items}</ul>`;
  });
  // Newlines to <br />
  htmlText = htmlText.replace(/\n/g, '<br />');
  // Clean up extra <br /> tags that might appear before or after list items due to the above replacement
  htmlText = htmlText.replace(/<br \/><li>/g, '<li>');
  htmlText = htmlText.replace(/<\/li><br \/>/g, '</li>');
  return { __html: htmlText };
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [documentCount, setDocumentCount] = useState<number | null>(null); // State for document count
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [conversationHistory, setConversationHistory] = useState<HistoryItem[]>([]);
  const [darkMode, setDarkMode] = useState<boolean>(() => {
    // Initialize dark mode from localStorage or default to false
    const savedMode = localStorage.getItem('darkMode');
    try {
      return savedMode ? JSON.parse(savedMode) : false;
    } catch (e) {
      console.error("Error parsing dark mode from localStorage, defaulting to false:", e);
      return false;
    }
  });

  useEffect(() => {
    // Apply dark mode class to body
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    // Save dark mode preference to localStorage
    localStorage.setItem('darkMode', JSON.stringify(darkMode)); // Corrected line
  }, [darkMode]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Initial welcome message
    setMessages([
      {
        id: 1,
        text: "Hello! I'm here to help answer your medical questions and provide health information. Please remember that I'm not a substitute for professional medical advice, diagnosis, or treatment. How can I assist you today?",
        sender: 'assistant',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);

    // Fetch document count from the backend
    const fetchDocumentCount = async () => {
      try {
        const response = await fetch(`${API_URL}/document_count`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        setDocumentCount(data.count);
      } catch (error) {
        console.error('Error fetching document count:', error);
        setDocumentCount(0); // Set to 0 or null if there's an error
      }
    };
    fetchDocumentCount();
  }, []); // Empty dependency array means this runs once on component mount

  const sendMessage = async () => {
    if (input.trim() === '') return; // Don't send empty messages

    const userMessage: Message = {
      id: Date.now(),
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Add user message to display immediately
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    // Prepare history for the backend request
    const historyForRequest: HistoryItem[] = [...conversationHistory, { isUser: true, content: userMessage.text }];
    setInput(''); // Clear input field
    setIsLoading(true); // Show loading indicator

    const assistantMessageId = Date.now() + 1;
    let assistantResponseText = '';
    let sourcesReceived: Source[] = [];

    // Add a placeholder for the assistant's response
    setMessages((prev) => [...prev, {
      id: assistantMessageId,
      text: '...', // Placeholder text
      sender: 'assistant',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    }]);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userMessage.text, history: historyForRequest }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { value, done } = await reader.read();
        if (done) break; // Stream finished

        const chunk = decoder.decode(value, { stream: true });
        // Split by 'data: ' and filter out empty strings
        const lines = chunk.split('\n\n').filter(line => line.startsWith('data: '));

        for (const line of lines) {
          try {
            const jsonString = line.substring(6); // Remove 'data: ' prefix
            const data = JSON.parse(jsonString);

            if (data.text) {
              assistantResponseText += data.text;
            }
            if (data.sources) {
              sourcesReceived = data.sources; // Update sources as they come in
            }
            if (data.error) {
              assistantResponseText = data.error;
              reader.cancel(); // Stop reading on error
              break;
            }

            // Update the message with streaming text and sources
            setMessages((prev) =>
              prev.map((msg) => 
                msg.id === assistantMessageId ? {
                  ...msg,
                  text: assistantResponseText + (sourcesReceived.length > 0 ? `\n\n**Sources:**\n${sourcesReceived.map(s => `- [${s.filename}](${s.url})`).join('\n')}` : '')
                } : msg
              )
            );
          } catch (parseError) {
            console.error('Error parsing stream chunk:', parseError, 'Chunk:', line);
          }
        }
      }
      // After the stream finishes, update conversation history with the full response
      const finalAssistantText = assistantResponseText + (sourcesReceived.length > 0 ? `\n\n**Sources:**\n${sourcesReceived.map(s => `- [${s.filename}](${s.url})`).join('\n')}` : '');
      setConversationHistory([...historyForRequest, { isUser: false, content: finalAssistantText }]);

    } catch (error) {
      console.error('Error sending message or processing stream:', error);
      // Display a generic error message to the user
      setMessages((prev) => prev.map((msg) => msg.id === assistantMessageId ? { ...msg, text: 'A network error occurred or the request failed. Please try again.' } : msg ));
    } finally {
      setIsLoading(false); // Hide loading indicator
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !isLoading) {
      sendMessage();
    }
  };

  return (
    <div className={`flex flex-col h-screen ${darkMode ? 'bg-gray-900 text-gray-100' : 'bg-gradient-to-br from-blue-50 to-purple-50'} antialiased transition-colors duration-300`}>
      <div className={`relative flex flex-col flex-1 max-w-5xl w-full mx-auto my-2 rounded-3xl shadow-2xl overflow-hidden ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
        <header className={`flex items-center p-4 ${darkMode ? 'bg-gray-700' : 'bg-gradient-to-r from-blue-600 to-purple-600'} text-white shadow-lg rounded-t-3xl`}>
          <img src={chlaLogo} alt="CHLA Logo" className="h-12 w-12 mr-4 rounded-full object-contain bg-white p-1 shadow-inner" />
          <div className="flex-1">
            <h1 className="text-xl font-bold">CHLA Virtual Health Librarian</h1>
            <p className="text-sm font-light opacity-90">Children's Hospital Los Angeles</p>
            <p className={`text-xs mt-1 font-semibold ${darkMode ? 'text-gray-400' : 'text-blue-200'}`}>
              Created by LMU's CHLA Capstone Team: Sam Biner, Jake Lestyk, Wanqiu Zhang, Jeenal Choudhary
            </p>
          </div>
          {/* Dark Mode Toggle */}
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 px-4 rounded-full transition-colors duration-300 ${darkMode ? 'bg-gray-600 hover:bg-gray-500' : 'bg-blue-500 hover:bg-blue-700'} text-white shadow-md flex items-center justify-center text-sm`}
            aria-label="Toggle dark mode"
          >
            Dark Mode
          </button>
        </header>

        <div className={`flex-1 overflow-y-auto p-4 space-y-4 ${darkMode ? 'bg-gray-800' : 'bg-gray-50'}`}>
          {messages.map((message) => (
            // Only render messages that are not the initial '...' placeholder
            message.text !== '...' && (
              <div key={message.id} className={`flex ${message.sender ==='user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-[75%] p-3 rounded-xl shadow-md ${message.sender === 'user' ? 'bg-blue-500 text-white rounded-br-none' : (darkMode ? 'bg-gray-700 text-gray-100' : 'bg-white text-gray-800') } rounded-bl-none border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}>
                  {/* Render markdown content safely */}
                  <p className="text-sm leading-relaxed" dangerouslySetInnerHTML={renderMarkdown(message.text)}></p>
                  <span className={`text-xs mt-1 block ${message.sender === 'user' ? 'text-blue-200' : (darkMode ? 'text-gray-400' : 'text-gray-500')} text-right`}>
                    {message.timestamp}
                  </span>
                </div>
              </div>
            )
          ))}
          {isLoading && (
            <div className="flex justify-start">
              {/* This div now matches the assistant's message bubble styling */}
              <div className={`max-w-[75%] p-3 rounded-xl shadow-md ${darkMode ? 'bg-gray-700 text-gray-100' : 'bg-white text-gray-800'} rounded-bl-none border ${darkMode ? 'border-gray-600' : 'border-gray-200'}`}>
                <div className="flex items-center space-x-2">
                  <div className="dot-flashing"></div>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} /> {/* Scroll target */}
        </div>

        <div className={`p-4 ${darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-100'} border-t shadow-inner`}>
          <div className="flex items-center space-x-3">
            <input
              type="text"
              className={`flex-1 p-3 border rounded-full focus:outline-none focus:ring-2 shadow-sm text-sm ${darkMode ? 'bg-gray-700 border-gray-600 text-gray-100 placeholder-gray-400 focus:ring-blue-400' : 'bg-white border-gray-300 text-gray-700 placeholder-gray-400 focus:ring-blue-500'}`}
              placeholder="Ask me about health topics, symptoms, or medical questions..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              className={`p-3 rounded-full ${darkMode ? 'bg-blue-700 hover:bg-blue-600' : 'bg-blue-600 hover:bg-blue-700'} text-white flex items-center justify-center shadow-lg transition-all duration-200 ${isLoading ? 'opacity-50 cursor-not-allowed' : 'active:scale-95'}`}
              disabled={isLoading}
            >
              <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5"/>
              </svg>
            </button>
          </div>
        </div>

        <footer className={`w-full p-3 text-center text-xs ${darkMode ? 'bg-gray-700 text-gray-300 border-gray-600' : 'bg-gray-100 text-gray-600 border-gray-200'} border-t rounded-b-3xl`}>
          <p>
            This AI assistant provides general health information and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have.
            {documentCount !== null && (
              <span className="block mt-1 text-gray-500">
                Knowledge base includes {documentCount} documents.
              </span>
            )}
          </p>
          <p className="mt-2 text-xs">
            If you have questions about our library, contact us here:
            <span className={`block font-semibold ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>Health Library: 323-361-2254</span>
          </p>
        </footer>
      </div>

      {/* Inline styles for the flashing dots animation */}
      <style>{`
        /* Add a root class for dark mode to apply base styles */
        html.dark {
          background-color: #1a202c; /* Dark background for the whole page */
          color: #e2e8f0; /* Light text color */
        }

        .dot-flashing {
          position: relative;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          background-color: #9ca3af;
          color: #9ca3af;
          animation: dotFlashing 1s infinite linear alternate;
          animation-delay: 0s;
        }

        .dot-flashing::before, .dot-flashing::after {
        content: '';
        display: inline-block;
        position: absolute;
        top: 0;
        }

        .dot-flashing::before {
        left: -15px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #9ca3af;
        color: #9ca3af;
        animation: dotFlashing 1s infinite alternate;
        animation-delay: 0s;
        }

        .dot-flashing::after {
        left: 15px;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #9ca3af;
        color: #9ca3af;
        animation: dotFlashing 1s infinite alternate;
        animation-delay: 1s;
        }

        @keyframes dotFlashing {
          0% {
            background-color: #9ca3af;
          }
          50%,
          100% {
            background-color: #e5e7eb;
          }
        }
      `}</style>
    </div>
  );
}
