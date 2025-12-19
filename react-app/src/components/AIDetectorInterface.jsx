import React, { useState } from 'react';
import { Send, AlertCircle, CheckCircle, Loader, BarChart3, Brain, Zap } from 'lucide-react';

const AIDetectorInterface = () => {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Simula l'analisi (in produzione chiameresti la tua API Python)
  const analyzeText = async () => {
    if (text.trim().length < 50) {
      alert('⚠️ Inserisci almeno 50 caratteri!');
      return;
    }

    setIsAnalyzing(true);
    
    // Simula chiamata API
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Calcola metriche simulate basate sul testo
    const sentences = text.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const words = text.toLowerCase().match(/\b\w+\b/g) || [];
    const uniqueWords = new Set(words);
    
    // Metriche simulate
    const avgSentenceLength = words.length / sentences.length;
    const lexicalDiversity = uniqueWords.size / words.length;
    const repetitiveness = 1 - lexicalDiversity;
    const burstiness = Math.random() * 5 + 2;
    
    // Determina se è AI (logica semplificata per demo)
    const aiScore = (
      (repetitiveness > 0.4 ? 30 : 10) +
      (avgSentenceLength > 15 && avgSentenceLength < 25 ? 25 : 10) +
      (burstiness < 4 ? 25 : 10) +
      Math.random() * 20
    );
    
    const isAI = aiScore > 50;
    const confidence = isAI ? aiScore : (100 - aiScore);
    
    setResult({
      isAI,
      confidence: Math.min(95, confidence),
      metrics: {
        repetitiveness: repetitiveness * 100,
        lexicalDiversity: lexicalDiversity * 100,
        burstiness: burstiness,
        avgSentenceLength: avgSentenceLength,
        readability: 60 + Math.random() * 30,
        templateBias: isAI ? 70 + Math.random() * 20 : 30 + Math.random() * 20
      },
      topIndicators: isAI ? [
        { name: 'Ripetitività Strutturale', score: repetitiveness * 100, icon: '🔄' },
        { name: 'Uso di Template', score: 75 + Math.random() * 15, icon: '📋' },
        { name: 'Bassa Variazione', score: (5 - burstiness) * 20, icon: '📉' }
      ] : [
        { name: 'Alta Variazione', score: burstiness * 15, icon: '🎨' },
        { name: 'Ricchezza Lessicale', score: lexicalDiversity * 100, icon: '📚' },
        { name: 'Stile Naturale', score: 70 + Math.random() * 20, icon: '✍️' }
      ]
    });
    
    setIsAnalyzing(false);
  };

  const exampleTexts = {
    ai: "Artificial intelligence represents a significant advancement in modern technology. It enables machines to perform tasks that typically require human intelligence. Machine learning algorithms analyze vast amounts of data to identify patterns. These systems continue to evolve and improve over time.",
    human: "So I was thinking about this the other day... AI is pretty wild, right? Like, sometimes it's spot on, but other times it says the weirdest stuff. Makes you wonder what's really going on in those neural networks. Anyway, I'm still trying to wrap my head around how it all works!"
  };

  return (
    <div className="main-wrapper">
      <div className="header-section">
        <div className="title-container">
          <Brain className="main-title" style={{width: '48px', height: '48px', WebkitTextFillColor: 'initial'}} />
          <h1 className="main-title">AI Detector</h1>
        </div>
        <p className="subtitle">Analizza qualsiasi testo per determinare se è stato generato da AI</p>
      </div>

      <div className="app-grid">
        {/* INPUT */}
        <div className="glass-card">
          <h2 className="card-title"><Zap style={{color: '#facc15'}} /> Inserisci Testo</h2>
          <textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Incolla qui il testo da analizzare..."
            className="text-input"
          />
          <div style={{display: 'flex', justifyContent: 'space-between', marginTop: '1rem', alignItems: 'center'}}>
            <span style={{color: text.length < 50 ? '#f87171' : '#4ade80', fontSize: '0.875rem'}}>
              {text.length} caratteri {text.length < 50 && '(minimo 50)'}
            </span>
            <button className="btn-primary" onClick={analyzeText} disabled={isAnalyzing || text.length < 50}>
              {isAnalyzing ? <Loader className="animate-spin" /> : <Send size={18} />}
              {isAnalyzing ? 'Analisi...' : 'Analizza'}
            </button>
          </div>
          <div style={{marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px solid rgba(168,85,247,0.2)'}}>
             <p style={{fontSize: '0.875rem', color: '#64748b', marginBottom: '0.5rem'}}>Prova un esempio:</p>
             <div style={{display: 'flex', gap: '0.5rem'}}>
                <button onClick={() => setText(exampleTexts.ai)} className="example-btn btn-ai">🤖 Testo AI</button>
                <button onClick={() => setText(exampleTexts.human)} className="example-btn btn-human">✍️ Testo Umano</button>
             </div>
          </div>
        </div>

        {/* RISULTATI */}
        <div className="glass-card">
          <h2 className="card-title"><BarChart3 style={{color: '#a855f7'}} /> Risultati Analisi</h2>
          {!result && !isAnalyzing && (
            <div style={{textAlign: 'center', color: '#64748b', marginTop: '4rem'}}>
              <AlertCircle style={{width: '64px', height: '64px', margin: '0 auto 1rem', opacity: 0.5}} />
              <p>Inserisci un testo e clicca "Analizza"</p>
            </div>
          )}
          
          {result && !isAnalyzing && (
            <div className="fade-in">
              <div className={`result-box ${result.isAI ? 'ai-result' : 'human-result'}`}>
                <div style={{display: 'flex', gap: '1rem', alignItems: 'center'}}>
                  {result.isAI ? <AlertCircle color="#f87171"/> : <CheckCircle color="#4ade80"/>}
                  <div>
                    <h3 style={{fontSize: '1.5rem', fontWeight: 'bold'}}>{result.isAI ? '🤖 GENERATO DA AI' : '✍️ SCRITTO DA UMANO'}</h3>
                    <p style={{fontSize: '0.875rem', opacity: 0.7}}>Confidenza: {result.confidence.toFixed(1)}%</p>
                  </div>
                </div>
                <div className="confidence-bg">
                  <div className="confidence-fill" style={{width: `${result.confidence}%`, background: result.isAI ? '#ef4444' : '#22c55e'}} />
                </div>
              </div>
              
              {/* Qui puoi aggiungere le altre metriche con stili simili */}
            </div>
          )}
        </div>
      </div>
      
      <div style={{marginTop: '2rem', textAlign: 'center', color: '#64748b', fontSize: '0.875rem'}}>
        <p>🎓 Progetto AI Detector • Modello Hybrid (BERT + Stylometric Features)</p>
      </div>
    </div>
  );
};

export default AIDetectorInterface;