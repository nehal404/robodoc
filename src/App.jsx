import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import jsyaml from 'js-yaml';

// AIProcessor Component
const AIProcessor = {
  async loadModelAndLabels(segment) {
    const segmentAssets = {
      eye: { model: '/models/eye_model/model.json', yaml: '/models/eye_model/metadata.yaml' },
      ear: { model: '/models/ear_model/model.json', yaml: '/models/ear_model/metadata.yaml' },
      skin: { model: '/models/skin_model/model.json', yaml: '/models/skin_model/metadata.yaml' },
      scalp: { model: '/models/scalp_model/model.json', yaml: '/models/scalp_model/metadata.yaml' },
      teeth: { model: '/models/oral_model/model.json', yaml: '/models/oral_model/metadata.yaml' },
    };
    if (!segmentAssets[segment]) throw new Error(`Invalid segment: ${segment}`);
    const assets = segmentAssets[segment];

    const yamlResponse = await fetch(assets.yaml);
    if (!yamlResponse.ok) throw new Error(`Failed to load YAML for ${segment}: ${yamlResponse.status}`);
    const yamlText = await yamlResponse.text();
    const yamlData = jsyaml.load(yamlText);
    const classNames = Object.values(yamlData.names);

    const model = await tf.loadGraphModel(assets.model);
    return { model, classNames };
  },

  async preprocessImage(imageFile, segment) {
    const resolutions = {
      skin: { width: 320, height: 320 },
      scalp: { width: 320, height: 320 },
      eye: { width: 416, height: 416 },
      ear: { width: 416, height: 416 },
      teeth: { width: 416, height: 416 },
    };
    const { width, height } = resolutions[segment] || { width: 416, height: 416 };

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.src = URL.createObjectURL(imageFile);
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, width, height);
        const imageData = ctx.getImageData(0, 0, width, height).data;
        const input = new Float32Array(width * height * 3);
        let pixelIndex = 0;
        for (let i = 0; i < imageData.length; i += 4) {
          input[pixelIndex++] = imageData[i] / 255.0;
          input[pixelIndex++] = imageData[i + 1] / 255.0;
          input[pixelIndex++] = imageData[i + 2] / 255.0;
        }
        resolve(input);
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => reject(new Error('Failed to load image'));
    });
  },

  async processImage(imageFile, segment, setResult, setProcessing, setImageUrl) {
    setProcessing(true);
    setResult('Processing...');
    try {
      const { model, classNames } = await this.loadModelAndLabels(segment);
      const input = await this.preprocessImage(imageFile, segment);
      const resolutions = {
        skin: { width: 320, height: 320 },
        scalp: { width: 320, height: 320 },
        eye: { width: 416, height: 416 },
        ear: { width: 416, height: 416 },
        teeth: { width: 416, height: 416 },
      };
      const { width, height } = resolutions[segment] || { width: 416, height: 416 };
      const tensor = tf.tensor4d(input, [1, height, width, 3]);
      const output = await model.executeAsync(tensor);
      const outputData = await output.data();
      const predictedClassIdx = outputData.indexOf(Math.max(...outputData));
      const confidence = outputData[predictedClassIdx];
      setResult(`Diagnosis for ${segment}: ${classNames[predictedClassIdx]} (Confidence: ${(confidence * 100).toFixed(2)}%)`);
      setImageUrl(URL.createObjectURL(imageFile));
      tf.dispose([tensor, output, model]);
    } catch (e) {
      setResult(`Error: ${e.message}`);
    } finally {
      setProcessing(false);
    }
  },
};

// GroqClient
const GroqClient = {
  async createChatCompletion(messages, apiKey) {
    try {
      const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json; charset=UTF-8',
          'Accept': 'application/json; charset=UTF-8',
        },
        body: JSON.stringify({
          messages,
          model: 'meta-llama/llama-4-maverick-17b-128e-instruct',
          temperature: 0.5,
          max_tokens: 1024,
        }),
      });
      if (response.ok) {
        return await response.json();
      }
      throw new Error(`API Error: ${response.status}`);
    } catch (e) {
      throw new Error(`Network Error: ${e.message}`);
    }
  },
};

// HomePage Component
const HomePage = ({ setPage }) => (
  <div className="container mx-auto p-4">
    <header className="flex items-center justify-between bg-blue-600 text-white p-4 rounded-lg">
      <div className="flex items-center">
        <img src="/assets/logo.png" alt="Logo" className="h-12 mr-4" />
        <h1 className="text-2xl font-bold">RoboDoc</h1>
      </div>
      <img src="/assets/lab_logo.png" alt="Lab Logo" className="h-12 rounded-full" />
    </header>
    <div className="mt-6 space-y-6">
      <SectionCard
        title="Body Scan"
        imagePath="/assets/body_scan.png"
        icons={['👁️', '👂', '🖐️', '🗣️', '🩺', '💆']}
        labels={['Eye', 'Ear', 'Skin', 'Throat', '', 'Scalp']}
        onSelect={() => setPage('body_scan')}
      />
      <SectionCard
        title="RoboDoc Chat"
        imagePath="/assets/robo_doc_chat.png"
        onSelect={() => setPage('robo_doc_chat')}
      />
      <SectionCard
        title="Invest Checkup"
        imagePath="/assets/invest_checkup.png"
        onSelect={() => setPage('invest_checkup')}
      />
    </div>
    <nav className="fixed bottom-0 left-0 right-0 bg-white shadow-lg flex justify-around py-2">
      {['🏠', '📊', '💬', '⚙️'].map((icon, index) => (
        <button key={index} className="text-2xl">{icon}</button>
      ))}
    </nav>
  </div>
);

// SectionCard Component
const SectionCard = ({ title, imagePath, icons, labels, onSelect }) => (
  <div className="bg-white p-6 rounded-lg shadow-md border border-blue-500 flex items-center">
    <div className="flex-1">
      <h2 className="text-xl font-bold">{title}</h2>
      {icons && labels && (
        <div className="flex flex-wrap gap-4 mt-2">
          {icons.map((icon, index) => (
            <div key={index} className="text-center">
              <span className="text-2xl">{icon}</span>
              <p className="text-sm">{labels[index]}</p>
            </div>
          ))}
        </div>
      )}
      <button
        onClick={onSelect}
        className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-full"
      >
        Select
      </button>
    </div>
    <img src={imagePath} alt={title} className="w-24 h-24 rounded-lg" />
  </div>
);

// BodyScan Component
const BodyScan = ({ setPage }) => {
  const [selectedModel, setSelectedModel] = useState(null);
  const [result, setResult] = useState('');
  const [processing, setProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const fileInputRef = useRef(null);

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file || !selectedModel) {
      setResult('Please select a segment before uploading an image');
      return;
    }
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setResult('Unsupported image format. Please use JPEG or PNG.');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setResult('Image too large. Please use an image smaller than 5MB.');
      return;
    }
    await AIProcessor.processImage(file, selectedModel, setResult, setProcessing, setImageUrl);
  };

  const handleDownloadReport = () => {
    if (!result || result.includes('Error') || processing) {
      setResult('No valid diagnosis available to download');
      return;
    }
    const blob = new Blob([result], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `diagnosis_report_${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const checkItems = [
    { title: 'Eyes Check', segment: 'eye', image: '/assets/eye.png' },
    { title: 'Ears Check', segment: 'ear', image: '/assets/ear.png' },
    { title: 'Skin Check', segment: 'skin', image: '/assets/skin.png' },
    { title: 'Scalp Check', segment: 'scalp', image: '/assets/scalp.png' },
    { title: 'Teeth Check', segment: 'teeth', image: '/assets/teeth.png' },
  ];

  return (
    <div className="container mx-auto p-4">
      <header className="flex items-center justify-between bg-blue-600 text-white p-4 rounded-lg">
        <button onClick={() => setPage('home')} className="text-white">← Back</button>
        <h1 className="text-2xl font-bold">Body Scan</h1>
        <div></div>
      </header>
      <div className="mt-6 space-y-4">
        <div className="flex justify-around">
          <button
            onClick={() => fileInputRef.current.click()}
            disabled={processing}
            className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          >
            Upload Image
          </button>
          <input
            type="file"
            accept="image/*"
            ref={fileInputRef}
            onChange={handleImageUpload}
            className="hidden"
          />
          <button
            onClick={() => alert('Camera capture not supported in browser')}
            disabled={processing}
            className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
          >
            Capture Image
          </button>
        </div>
        {imageUrl && (
          <img src={imageUrl} alt="Selected" className="mt-4 mx-auto h-48 object-contain" />
        )}
        <div className="space-y-2">
          {checkItems.map((item) => (
            <div key={item.segment} className="flex items-center p-2 bg-white rounded shadow">
              <input
                type="radio"
                name="segment"
                value={item.segment}
                checked={selectedModel === item.segment}
                onChange={() => {
                  setSelectedModel(item.segment);
                  setResult(`Selected ${item.segment} for analysis`);
                }}
                disabled={processing}
                className="mr-2"
              />
              <span className="flex-1">{item.title}</span>
              <img src={item.image} alt={item.title} className="w-12 h-12" />
            </div>
          ))}
        </div>
        <p className="text-center text-lg">{result}</p>
        <button
          onClick={handleDownloadReport}
          disabled={processing}
          className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50 w-full"
        >
          Download Report
        </button>
      </div>
    </div>
  );
};

// RoboDocChat Component
const RoboDocChat = ({ setPage }) => {
  const [messages, setMessages] = useState([
    { sender: 'bot', text: 'Welcome to RoboDoc! Start typing in English or Arabic, and I’ll respond accordingly.' },
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const detectLanguage = (text) => {
    return /[\u0600-\u06FF]/.test(text) ? 'ar' : 'en';
  };

  const sendMessage = async () => {
    if (!input.trim()) return;
    const newMessages = [...messages, { sender: 'user', text: input }];
    setMessages(newMessages);
    setLoading(true);
    setInput('');

    const language = detectLanguage(input);
    try {
      const response = await GroqClient.createChatCompletion(
        [
          {
            role: 'system',
            content: 'You are named Robodoc. Please respond to all queries as if you are Robodoc, a knowledgeable and helpful assistant, keep your responses very short but informative. If the input is in Arabic, respond in Egyptian Arabic dialect.',
          },
          { role: 'user', content: input },
        ],
        import.meta.env.VITE_GROQ_API_KEY
      );
      setMessages([...newMessages, { sender: 'bot', text: response.choices[0].message.content }]);
    } catch (e) {
      setMessages([...newMessages, { sender: 'bot', text: language === 'en' ? `Sorry, something went wrong: ${e.message}` : `عذرا، حدث خطأ: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
    scrollToBottom();
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <div className="container mx-auto p-4">
      <header className="flex items-center justify-between bg-blue-600 text-white p-4 rounded-lg">
        <button onClick={() => setPage('home')} className="text-white">← Back</button>
        <h1 className="text-2xl font-bold">RoboDoc Chat</h1>
        <div></div>
      </header>
      <div className="mt-6 bg-white rounded-lg shadow p-4 h-[70vh] overflow-y-auto">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`mb-4 p-3 rounded-lg ${msg.sender === 'bot' ? 'bg-gray-200 mr-8' : 'bg-blue-100 ml-8'}`}
          >
            {msg.text}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      {loading && <div className="text-center my-4">Loading...</div>}
      <div className="mt-4 flex items-center">
        <button className="text-2xl mr-2">🎤</button>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="Type here..."
          className="flex-1 p-2 border rounded"
        />
        <button onClick={sendMessage} className="ml-2 text-2xl">➡️</button>
      </div>
    </div>
  );
};

// InvestCheckup Component
const InvestCheckup = ({ setPage }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [name, setName] = useState('');
  const [age, setAge] = useState('');
  const [result, setResult] = useState('');
  const [processing, setProcessing] = useState(false);
  const [imageUrl, setImageUrl] = useState(null);
  const fileInputRef = useRef(null);

  const stepSegments = {
    1: 'skin',
    2: 'scalp',
    3: 'eye',
    4: 'ear',
    5: 'teeth',
  };

  const handleImageUpload = async (e) => {
    const file = e.target.files[0];
    if (!file || !stepSegments[currentStep]) {
      setResult('Please select a valid step');
      return;
    }
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
      setResult('Unsupported image format. Please use JPEG or PNG.');
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setResult('Image too large. Please use an image smaller than 5MB.');
      return;
    }
    await AIProcessor.processImage(file, stepSegments[currentStep], setResult, setProcessing, setImageUrl);
  };

  const nextStep = () => {
    if (currentStep < 6) {
      setCurrentStep(currentStep + 1);
      setResult('');
      setImageUrl(null);
    }
  };

  const previousStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
      setResult('');
      setImageUrl(null);
    }
  };

  return (
    <div className="container mx-auto p-4">
      <header className="flex items-center justify-between bg-blue-600 text-white p-4 rounded-lg">
        <button onClick={() => setPage('home')} className="text-white">← Back</button>
        <h1 className="text-2xl font-bold">Invest Checkup</h1>
        <div></div>
      </header>
      <div className="mt-6 space-y-4">
        {currentStep === 0 && (
          <div>
            <h2 className="text-xl font-bold">Step 1: Patient Info</h2>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Name"
              className="w-full p-2 border rounded mt-4"
            />
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              placeholder="Age"
              className="w-full p-2 border rounded mt-4"
            />
          </div>
        )}
        {[1, 2, 3, 4, 5].includes(currentStep) && (
          <div>
            <h2 className="text-xl font-bold">
              Step {currentStep + 1}: {stepSegments[currentStep].charAt(0).toUpperCase() + stepSegments[currentStep].slice(1)} Health
            </h2>
            <div className="flex justify-around mt-4">
              <button
                onClick={() => fileInputRef.current.click()}
                disabled={processing}
                className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
              >
                Upload Image
              </button>
              <input
                type="file"
                accept="image/*"
                ref={fileInputRef}
                onChange={handleImageUpload}
                className="hidden"
              />
              <button
                onClick={() => alert('Camera capture not supported in browser')}
                disabled={processing}
                className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
              >
                Capture Image
              </button>
            </div>
            {imageUrl && (
              <img src={imageUrl} alt="Selected" className="mt-4 mx-auto h-48 object-contain" />
            )}
            {result && <p className="text-center text-lg mt-4">{result}</p>}
          </div>
        )}
        {currentStep === 6 && (
          <div>
            <h2 className="text-xl font-bold">Final Report</h2>
            <p className="mt-2">Overall Health Score: B</p>
            <p className="mt-2">Recommendations: Maintain a balanced diet and regular exercise.</p>
            <button
              onClick={() => {
                const blob = new Blob(['Overall Health Score: B\nRecommendations: Maintain a balanced diet and regular exercise.'], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `final_report_${Date.now()}.txt`;
                a.click();
                URL.revokeObjectURL(url);
              }}
              className="mt-4 bg-blue-600 text-white px-4 py-2 rounded w-full"
            >
              Download Report
            </button>
          </div>
        )}
        <div className="flex justify-between mt-4">
          {currentStep > 0 && (
            <button
              onClick={previousStep}
              disabled={processing}
              className="bg-gray-600 text-white px-4 py-2 rounded disabled:opacity-50"
            >
              Previous
            </button>
          )}
          {currentStep < 6 && (
            <button
              onClick={nextStep}
              disabled={processing}
              className="bg-blue-600 text-white px-4 py-2 rounded disabled:opacity-50"
            >
              Next
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

// Main App Component
const App = () => {
  const [page, setPage] = useState('home');

  return (
    <div>
      {page === 'home' && <HomePage setPage={setPage} />}
      {page === 'body_scan' && <BodyScan setPage={setPage} />}
      {page === 'robo_doc_chat' && <RoboDocChat setPage={setPage} />}
      {page === 'invest_checkup' && <InvestCheckup setPage={setPage} />}
    </div>
  );
};

export default App;