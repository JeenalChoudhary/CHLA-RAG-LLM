import React, { useState, useEffect, useRef } from 'react';
// Corrected import path: logo is in src/assets/, so use new URL() for robustness
const chlaLogo = new URL('./assets/image_copy.png', import.meta.url).href;

// Define the structure for a message
interface Message {
  id: number;
  text: string;
  sender: 'user' | 'assistant';
  timestamp: string;
}

// Helper function to convert basic Markdown to HTML
const renderMarkdown = (markdownText: string) => {
  let htmlText = markdownText;

  // Convert bold text (e.g., **text**)
  htmlText = htmlText.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // Convert list items (e.g., * item)
  const lines = htmlText.split('\n');
  let inList = false;
  let processedLines: string[] = [];

  for (const line of lines) {
    // Check for Markdown list items
    if (line.trim().startsWith('* ')) {
      if (!inList) {
        processedLines.push('<ul>');
        inList = true;
      }
      // Remove the '* ' prefix and trim, then wrap in <li>
      processedLines.push(`<li>${line.trim().substring(2).trim()}</li>`);
    } else {
      if (inList) {
        processedLines.push('</ul>');
        inList = false;
      }
      processedLines.push(line);
    }
  }
  if (inList) { // Close list if it was open at the end
    processedLines.push('</ul>');
  }
  htmlText = processedLines.join('\n');

  // Convert Markdown links [text](url) into HTML <a> tags
  // This regex specifically targets the format [filename.pdf](http://...)
  // It captures the link text ($1) and the URL ($2)
  htmlText = htmlText.replace(/\[(.*?)\]\((.*?)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer" class="text-blue-600 hover:underline">$1</a>');


  // Replace single newlines with <br> for line breaks within paragraphs, but not within ul/li or a tags
  // This regex avoids replacing newlines that are immediately followed by <li>, </ul>, or <a (start of an anchor tag)
  htmlText = htmlText.replace(/\n(?!<ul|<\/ul>|<li>|<a)/g, '<br/>');


  return { __html: htmlText };
};


export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [documentCount, setDocumentCount] = useState<number | null>(null); // State for document count
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // State to hold the conversation history for sending to the backend
  // The backend expects history as a list of {role: string, content: string} objects
  const [conversationHistory, setConversationHistory] = useState<
    { role: string; content: string }[]
  >([]);

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
        id: 1,
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
        setDocumentCount(0); // Set to 0 or some error indicator on failure
      }
    };

    fetchDocumentCount();
  }, []); // Empty dependency array means this runs once on mount

  const sendMessage = async () => {
    if (input.trim() === '') return;

    const newUserMessage: Message = {
      id: messages.length + 1,
      text: input,
      sender: 'user',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Add user message to display immediately
    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    // Add user message to history for backend
    setConversationHistory((prevHistory) => [
      ...prevHistory,
      { role: 'user', content: newUserMessage.text },
    ]);
    setInput('');
    setIsLoading(true);

    let assistantResponseText = '';
    const newAssistantMessage: Message = {
      id: messages.length + 2,
      text: '', // Will be updated as chunks arrive
      sender: 'assistant',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    };

    // Add a placeholder assistant message to the UI
    setMessages((prevMessages) => [...prevMessages, newAssistantMessage]);

    try {
      // Make a POST request to the backend with a JSON body
      const response = await fetch('http://127.0.0.1:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: newUserMessage.text,
          history: conversationHistory, // Send the current conversation history
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      let errorOccurredInStream = false; // Flag to track stream errors
      let sourcesReceived: { filename: string; url: string }[] = []; // Array to collect sources with links

      // Function to process each chunk from the stream
      const processStream = async () => {
        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          // Split by 'data: ' to process each SSE event
          const lines = chunk.split('\n\n').filter(line => line.startsWith('data: '));

          for (const line of lines) {
            try {
              const jsonString = line.substring(6); // Remove 'data: ' prefix
              const data = JSON.parse(jsonString);

              if (data.text) {
                assistantResponseText += data.text;
                // Update the last message in state with new text and current sources
                setMessages((prevMessages) =>
                  prevMessages.map((msg, index) =>
                    index === prevMessages.length - 1
                      ? {
                          ...msg,
                          text: assistantResponseText +
                            (sourcesReceived.length > 0
                              ? `\n\n**Sources:**\n${sourcesReceived.map(s => `- [${s.filename}](${s.url})`).join('\n')}`
                              : '')
                        }
                      : msg
                  )
                );
              } else if (data.sources) {
                // Append sources to the collected sources
                sourcesReceived = [...sourcesReceived, ...data.sources];
                // Update the message with current text + sources (if any)
                setMessages((prevMessages) =>
                  prevMessages.map((msg, index) =>
                    index === prevMessages.length - 1
                      ? {
                          ...msg,
                          text: assistantResponseText +
                            (sourcesReceived.length > 0
                              ? `\n\n**Sources:**\n${sourcesReceived.map(s => `- [${s.filename}](${s.url})`).join('\n')}`
                              : '')
                        }
                      : msg
                  )
                );
              } else if (data.error) {
                // If backend sends an explicit error, display it
                assistantResponseText = data.error;
                errorOccurredInStream = true; // Set flag
                setMessages((prevMessages) =>
                  prevMessages.map((msg, index) =>
                    index === prevMessages.length - 1
                      ? { ...msg, text: assistantResponseText }
                      : msg
                  )
                );
                reader.cancel(); // Stop reading further chunks
                break; // Exit inner loop
              }
            } catch (parseError) {
              console.error('Error parsing stream chunk:', parseError, 'Chunk:', line);
              assistantResponseText = 'Failed to parse backend response.';
              errorOccurredInStream = true;
              setMessages((prevMessages) =>
                prevMessages.map((msg, index) =>
                  index === prevMessages.length - 1
                    ? { ...msg, text: assistantResponseText }
                    : msg
                )
              );
              reader.cancel(); // Stop reading further chunks
              break; // Exit inner loop
            }
          }
          if (errorOccurredInStream) break; // Exit outer loop if error occurred
        }
      };

      await processStream();

      // After stream finishes, if no text was received and no explicit error was set
      if (!errorOccurredInStream && assistantResponseText.trim() === '') {
        assistantResponseText = "I couldn't find relevant information for that query. Please try rephrasing or asking a more specific question.";
        setMessages((prevMessages) =>
          prevMessages.map((msg, index) =>
            index === prevMessages.length - 1
              ? { ...msg, text: assistantResponseText }
              : msg
          )
        );
      }

      // Add assistant's final response to history
      setConversationHistory((prevHistory) => [
        ...prevHistory,
        { role: 'assistant', content: assistantResponseText },
      ]);

    } catch (error) {
      console.error('Error sending message or processing stream:', error);
      // Fallback for network errors or unhandled stream errors
      setMessages((prevMessages) =>
        prevMessages.map((msg, index) =>
          index === prevMessages.length - 1
            ? { ...msg, text: 'A network error occurred or the request failed. Please try again.' }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'Enter' && !isLoading) {
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-purple-50 antialiased">
      {/* Centered Container for the entire chat interface */}
      {/* Increased max-w to 5xl and reduced vertical margin to my-2 */}
      <div className="relative flex flex-col flex-1 max-w-5xl w-full mx-auto my-2 rounded-3xl shadow-2xl overflow-hidden bg-white">
        {/* Header - Top Bar */}
        {/* Reduced vertical padding from p-6 to p-4, and logo size from h-16/w-16 to h-12/w-12 */}
        <header className="flex items-center p-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg rounded-t-3xl">
          <img src={chlaLogo} alt="CHLA Logo" className="h-12 w-12 mr-4 rounded-full object-contain bg-white p-1 shadow-inner" />
          <div>
            <h1 className="text-xl font-bold">CHLA Health Librarian</h1> {/* Changed title here */}
            <p className="text-sm font-light opacity-90">Children's Hospital Los Angeles</p> {/* Reduced text size */}
          </div>
          {/* Removed star and heart icons */}
        </header>

        {/* Chat Messages Area */}
        {/* Adjusted padding from p-6 to p-4 for more vertical space */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${
                message.sender === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              <div
                className={`max-w-[75%] p-3 rounded-xl shadow-md ${ /* Reduced padding from p-4 to p-3 */
                  message.sender === 'user'
                    ? 'bg-blue-500 text-white rounded-br-none'
                    : 'bg-white text-gray-800 rounded-bl-none border border-gray-200'
                }`}
              >
                <p className="text-sm leading-relaxed" dangerouslySetInnerHTML={renderMarkdown(message.text)}></p> {/* Render Markdown */}
                <span
                  className={`text-xs mt-1 block ${ /* Reduced margin-top from mt-2 to mt-1 */
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
              <div className="max-w-[75%] p-3 rounded-xl shadow-md bg-white rounded-bl-none border border-gray-200">
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
        {/* Reduced vertical padding from p-6 to p-4 */}
        <div className="p-4 bg-white border-t border-gray-100 shadow-inner">
          <div className="flex items-center space-x-3"> {/* Reduced space-x from 4 to 3 */}
            <input
              type="text"
              className="flex-1 p-3 border border-gray-300 rounded-full focus:outline-none focus:ring-2 focus:ring-blue-500 shadow-sm text-gray-700 placeholder-gray-400 text-sm" /* Reduced padding and text size */
              placeholder="Ask me about health topics, symptoms, or medical questions..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              disabled={isLoading}
            />
            <button
              onClick={sendMessage}
              className={`p-3 rounded-full bg-blue-600 text-white flex items-center justify-center shadow-lg transition-all duration-200 ${ /* Reduced padding */
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
                className="w-5 h-5" /* Reduced icon size */
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
        {/* Reduced vertical padding from p-4 to p-3, and text size to text-xs */}
        <footer className="w-full bg-gray-100 p-3 text-center text-xs text-gray-600 border-t border-gray-200 rounded-b-3xl">
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
