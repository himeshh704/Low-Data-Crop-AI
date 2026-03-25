"use client"
import { useState, useRef, useEffect } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { UploadCloud, Activity, Zap, CheckCircle2, AlertTriangle, ShieldAlert, Server, Brain } from "lucide-react"

export default function Home() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<any>(null)
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [health, setHealth] = useState<any>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    fetch("${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/health").then(r => r.json()).then(setHealth).catch(() => {})
    fetch("${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/model-info").then(r => r.json()).then(setModelInfo).catch(() => {})
  }, [])

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selected = e.target.files?.[0]
    if (selected) {
      setFile(selected)
      setPreview(URL.createObjectURL(selected))
      setResult(null)
    }
  }

  const analyzeImage = async () => {
    if (!file) return
    setIsAnalyzing(true)
    try {
      const formData = new FormData()
      formData.append("file", file)
      
      const res = await fetch("${process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000"}/predict", {
        method: "POST",
        body: formData,
      })
      if (!res.ok) throw new Error("API Error")
      setResult(await res.json())
    } catch (err) {
      alert("Failed to connect to AI backend. Make sure uvicorn is running on port 8000!")
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <main className="min-h-screen bg-black text-white selection:bg-emerald-500/30 flex flex-col items-center py-20 px-4 font-sans relative overflow-hidden">
      {/* Background gradients */}
      <div className="absolute top-0 -left-64 w-96 h-96 bg-emerald-500/20 rounded-full blur-[128px] opacity-50 select-none pointer-events-none" />
      <div className="absolute top-40 -right-64 w-96 h-96 bg-blue-500/20 rounded-full blur-[128px] opacity-50 select-none pointer-events-none" />
      
      <div className="z-10 text-center max-w-2xl mx-auto mb-12">
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-sm text-emerald-400 mb-6"
        >
          <Zap className="w-4 h-4" />
          Prototypical Few-Shot AI
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="text-5xl md:text-6xl font-bold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white via-white to-white/60"
        >
          Detect Crop Stress <br className="hidden md:block" /> with Extreme Precision
        </motion.h1>
        <motion.p 
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-lg text-zinc-400 leading-relaxed"
        >
          Upload a high-resolution leaf scan. Our self-learning AI calculates
          Euclidean distance across geometric class centroids to diagnose health.
        </motion.p>
      </div>

      <motion.div 
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.3 }}
        className="z-10 w-full max-w-xl bg-zinc-900/50 border border-white/10 rounded-3xl p-8 backdrop-blur-xl shadow-2xl relative"
      >
        <AnimatePresence mode="wait">
          {!preview ? (
            <motion.div 
              key="upload"
              exit={{ opacity: 0, scale: 0.95 }}
              onClick={() => fileInputRef.current?.click()}
              className="border-2 border-dashed border-white/10 rounded-2xl p-12 text-center cursor-pointer hover:border-emerald-500/50 hover:bg-emerald-500/5 transition-all group"
            >
              <input 
                type="file" 
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/*" 
                className="hidden" 
              />
              <UploadCloud className="w-12 h-12 text-zinc-500 mx-auto mb-4 group-hover:text-emerald-400 transition-colors" />
              <h3 className="text-xl font-medium text-white mb-2">Upload crop image</h3>
              <p className="text-sm text-zinc-500">Drag & drop or click to browse</p>
            </motion.div>
          ) : (
            <motion.div 
              key="preview"
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="flex flex-col items-center"
            >
              <div className="relative w-full aspect-video rounded-2xl overflow-hidden border border-white/10 mb-6 bg-black/50">
                <img src={preview} alt="Crop preview" className="object-cover w-full h-full" />
                <button 
                  onClick={() => { setPreview(null); setFile(null); setResult(null); }}
                  className="absolute top-4 right-4 bg-black/60 hover:bg-black/80 text-white text-xs px-3 py-1.5 rounded-full backdrop-blur-md transition-all border border-white/10"
                >
                  Discard
                </button>
              </div>

              {!result ? (
                <button
                  onClick={analyzeImage}
                  disabled={isAnalyzing}
                  className="w-full relative overflow-hidden bg-emerald-600 hover:bg-emerald-500 text-white font-medium py-4 px-6 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed group"
                >
                  {isAnalyzing ? (
                    <span className="flex items-center justify-center gap-3">
                      <Activity className="w-5 h-5 animate-pulse" />
                      Computing Prototypes...
                    </span>
                  ) : (
                     <span className="flex items-center justify-center gap-3">
                      <Zap className="w-5 h-5 group-hover:scale-110 transition-transform" />
                      Run Analysis Pipeline
                    </span>
                  )}
                </button>
              ) : (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="w-full bg-black/40 border border-white/10 rounded-2xl p-6"
                >
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-lg font-medium text-zinc-300">Diagnosis</h3>
                    <div className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2 ${
                      result.prediction === 'Healthy' ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30' :
                      result.prediction === 'Stressed' ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/30' :
                      'bg-red-500/20 text-red-400 border border-red-500/30'
                    }`}>
                      {result.prediction === 'Healthy' ? <CheckCircle2 className="w-4 h-4"/> : 
                       result.prediction === 'Stressed' ? <AlertTriangle className="w-4 h-4"/> : 
                       <ShieldAlert className="w-4 h-4"/>}
                      {result.prediction}
                    </div>
                  </div>

                  <div className="space-y-4">
                    <p className="text-sm text-zinc-500 mb-2">Confidence Matrix</p>
                    {Object.entries(result.all_probabilities).map(([cls, prob]: any) => (
                      <div key={cls} className="w-full">
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-zinc-300">{cls}</span>
                          <span className="font-mono text-zinc-400">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 w-full bg-white/5 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${prob * 100}%` }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            className={`h-full rounded-full ${
                              cls === 'Healthy' ? 'bg-emerald-500' :
                              cls === 'Stressed' ? 'bg-yellow-500' : 'bg-red-500'
                            }`}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </motion.div>
    </main>
  )
}
