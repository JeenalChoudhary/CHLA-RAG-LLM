import React, { useState, useRef, useEffect } from 'react';
import { Send, User, Bot, Heart, Stethoscope } from 'lucide-react';

interface Message {
  id: number;
  text: string;
  isUser: boolean;
  timestamp: Date;
}

const ButterflyIcon = () => (
  <img
     src="/image copy.png"
     alt="CHLA Butterfly Logo"
     className="w-6 h-6 object-contain"
  />
);

const CHLAMedicalAssistant: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 1,
      text: "Hello! I'm here to help answer your medical questions and provide health information. Please remember that I'm not a substitute for professional medical advice, diagnosis, or treatment. How can I assist you today?",
      isUser: false,
      timestamp: new Date(),
    },
  ]);
  const [inputText, setInputText] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // --- CRITICAL MODIFICATION: Connect to Flask Backend API ---
  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage: Message = {
      id: messages.length + 1,
      text: inputText,
      isUser: true,
      timestamp: new Date(),
    };
    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsTyping(true);

    // Prepare conversation history for the backend
    // Exclude the initial greeting message from the history sent to the backend
    const conversationHistoryForBackend = messages.slice(1).map(msg => ({
        text: msg.text,
        isUser: msg.isUser
    }));


    let aiResponseText = '';
    let collectedSources: string[] = [];
    let currentAIMessageId: number; // Will store the ID of the AI message being built

    try {
        // Add a placeholder AI message immediately to show typing indicator
        setMessages(prev => {
            const newAIMessage: Message = {
                id: prev.length + 1,
                text: '', // Start with empty text, will be updated by streaming
                isUser: false,
                timestamp: new Date(),
            };
            currentAIMessageId = newAIMessage.id; // Capture the ID for subsequent updates
            return [...prev, newAIMessage];
        });

        const response = await fetch('http://localhost:5000/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: userMessage.text,
                history: conversationHistoryForBackend
            }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder('utf-8');

        if (reader) {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                const chunk = decoder.decode(value, { stream: true });
                // Process Server-Sent Events (SSE)
                const lines = chunk.split('\n\n').filter(line => line.trim() !== '');

                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6); // Remove 'data: ' prefix
                        try {
                            const parsedData = JSON.parse(data);
                            if (parsedData.sources) {
                                collectedSources = parsedData.sources;
                            } else if (parsedData.error) {
                                // Handle error streamed from backend
                                aiResponseText += `Error: ${parsedData.error}`;
                                // Update the message in state
                                setMessages(prev =>
                                    prev.map(msg =>
                                        msg.id === currentAIMessageId
                                            ? { ...msg, text: aiResponseText }
                                            : msg
                                    )
                                );
                                throw new Error(parsedData.error);
                            }
                            // If it's sources or error, it's not text to append directly to aiResponseText
                            // The text is only appended if it's not a parsable JSON object for sources/error
                        } catch (parseError) {
                            // If it's not JSON, it's probably a text chunk
                            aiResponseText += data;
                        }
                        // Update the message in state
                        setMessages(prev =>
                            prev.map(msg =>
                                msg.id === currentAIMessageId
                                    ? { ...msg, text: aiResponseText }
                                    : msg
                            )
                        );
                    }
                }
            }
        }

        // After streaming, append sources if any
        if (collectedSources.length > 0) {
            const sourcesText = "\n\n**Sources:**\n" + collectedSources.map(s => `- ${s.replace(/_/g, " ").replace(/\.pdf$/i, "")}`).join("\n");
            setMessages(prev =>
                prev.map(msg =>
                    msg.id === currentAIMessageId
                        ? { ...msg, text: aiResponseText + sourcesText }
                        : msg
                )
            );
        }

    } catch (error: any) { // Use 'any' for error to handle different types
        console.error("Error sending message to backend:", error);
        setMessages(prev => {
            // Find the AI message that was being updated, or add a new one if it wasn't started
            const existingAIMessageIndex = prev.findIndex(msg => msg.id === currentAIMessageId && !msg.isUser);
            const errorMessageText = `Sorry, there was an error connecting to the AI or processing your request: ${error.message}. Please try again.`;

            if (existingAIMessageIndex !== -1) {
                return prev.map((msg, index) =>
                    index === existingAIMessageIndex
                        ? { ...msg, text: msg.text === '' ? errorMessageText : msg.text + "\n\n" + errorMessageText }
                        : msg
                );
            } else {
                const errorMessage: Message = {
                    id: prev.length + 1,
                    text: errorMessageText,
                    isUser: false,
                    timestamp: new Date(),
                };
                return [...prev, errorMessage];
            }
        });
    } finally {
        setIsTyping(false);
    }
  };
  // --- END CRITICAL MODIFICATION ---

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-blue-100">
        <div className="max-w-4xl mx-auto px-4 py-4">
          <div className="flex items-center space-x-3">
            <div className="flex items-center space-x-2">
              <div className="p-2 bg-blue-100 rounded-full">
                <ButterflyIcon />
              </div>
              <div className="flex items-center space-x-1">
                <Stethoscope className="w-5 h-5 text-blue-600" />
                <Heart className="w-4 h-4 text-red-500" />
              </div>
            </div>
            <div>
              <h1 className="text-xl font-semibold text-gray-800">CHLA Medical Assistant</h1>
              <p className="text-sm text-gray-600">Children's Hospital Los Angeles</p>
            </div>
          </div>
        </div>
      </header>
      {/* Chat Container */}
      <div className="max-w-4xl mx-auto px-4 py-6">
        <div className="bg-white rounded-2xl shadow-lg border border-blue-100 overflow-hidden">
          {/* Messages Area */}
          <div className="h-96 overflow-y-auto p-6 space-y-4 bg-gradient-to-b from-blue-50/30 to-white">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex items-start space-x-3 ${
                  message.isUser ? 'flex-row-reverse space-x-reverse' : ''
                }`}
              >
                <div
                  className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    message.isUser
                      ? 'bg-blue-500 text-white'
                      : 'bg-gradient-to-br from-blue-100 to-indigo-100 text-blue-600'
                  }`}
                >
                  {message.isUser ? <User className="w-4 h-4" /> : <ButterflyIcon />}
                </div>
                <div
                  className={`max-w-xs lg:max-w-md px-4 py-3 rounded-2xl ${
                    message.isUser
                      ? 'bg-blue-500 text-white rounded-br-md'
                      : 'bg-white text-gray-800 shadow-sm border border-blue-100 rounded-bl-md'
                  }`}
                >
                  <p className="text-sm leading-relaxed">{message.text}</p>
                  <p
                    className={`text-xs mt-2 ${
                      message.isUser ? 'text-blue-100' : 'text-gray-500'
                    }`}
                  >
                    {message.timestamp.toLocaleTimeString([], {
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                  </p>
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-blue-100 to-indigo-100 text-blue-600 flex items-center justify-center">
                  <ButterflyIcon />
                </div>
                <div className="bg-white text-gray-800 shadow-sm border border-blue-100 rounded-2xl rounded-bl-md px-4 py-3">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          {/* Input Area */}
          <div className="border-t border-blue-100 bg-white p-4">
            <div className="flex items-end space-x-3">
              <div className="flex-1">
                <textarea
                  value={inputText}
                  onChange={(e) => setInputText(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask me about health topics, symptoms, or medical questions..."
                  className="w-full px-4 py-3 border border-blue-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-blue-50/50"
                  rows={1}
                  style={{ minHeight: '44px', maxHeight: '120px' }}
                />
              </div>
              <button
                onClick={handleSendMessage}
                disabled={!inputText.trim()}
                className="flex-shrink-0 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed text-white p-3 rounded-xl transition-colors duration-200 shadow-sm hover:shadow-md"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
        {/* Disclaimer */}
        <div className="mt-4 text-center">
          <p className="text-xs text-gray-500 max-w-2xl mx-auto leading-relaxed">
            This AI assistant provides general health information and is not a substitute for professional medical advice,
             diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any
             questions you may have regarding a medical condition.
          </p>
          <div className="mt-3 pt-3 border-t border-gray-200">
            <p className="text-xs text-gray-500 mt-1">
              If you have any questions regarding our health library, please contact us here:
              <a href="tel:323-361-2254" className="text-blue-600 hover:text-blue-800 ml-1 font-medium">
                323-361-2254
              </a>
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CHLAMedicalAssistant;