import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  TrendingUp, TrendingDown, Activity, Zap, 
  RefreshCcw, Radio, Brain, Globe, Layers, PieChart
} from 'lucide-react';
import { 
  ResponsiveContainer, AreaChart, Area, Tooltip 
} from 'recharts';
import { motion } from 'framer-motion';

const Dashboard = () => {
  const [data, setData] = useState(null);
  const [backtest, setBacktest] = useState(null);
  const [loading, setLoading] = useState(true);
  const [market, setMarket] = useState('us');

  const fetchForecast = async (isInitial = false) => {
    if (isInitial) setLoading(true);
    try {
      const [fRes, bRes] = await Promise.all([
        axios.get(`https://shashank182123-sentfusion-api.hf.space/api/forecast?market=${market}`),
        axios.get(`https://shashank182123-sentfusion-api.hf.space/api/backtest?market=${market}`)
      ]);
      setData(fRes.data);
      setBacktest(bRes.data);
    } catch (error) {
      console.error("Error fetching data:", error);
    } finally {
      if (isInitial) setLoading(false);
    }
  };

  useEffect(() => {
    fetchForecast(true);
    const interval = setInterval(() => fetchForecast(false), 5000);
    return () => clearInterval(interval);
  }, [market]);

  if (loading || !data || !backtest) {
    return (
      <div className="min-h-screen bg-[#020617] flex items-center justify-center">
        <motion.div 
          animate={{ scale: [1, 1.1, 1], opacity: [0.5, 1, 0.5] }}
          transition={{ duration: 2, repeat: Infinity }}
          className="flex flex-col items-center gap-4"
        >
          <Brain className="w-12 h-12 text-cyan-500" />
          <p className="text-cyan-500 font-bold tracking-[0.2em] uppercase text-[10px]">Neural Syncing...</p>
        </motion.div>
      </div>
    );
  }

  const isBullish = data.signal === "BULLISH";

  return (
    <div className="min-h-screen relative pb-20 bg-[#020617] text-slate-200 font-['Inter'] selection:bg-cyan-500/30">
      <div className="mesh-bg" />
      
      {/* HEADER */}
      <header className="max-w-7xl mx-auto px-4 lg:px-8 pt-6 lg:pt-8 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-6">
        <div className="flex items-center gap-4">
          <div className="bg-cyan-600 p-2 rounded-xl shadow-lg shadow-cyan-900/20">
            <Brain className="w-5 h-5 lg:w-6 lg:h-6 text-black" />
          </div>
          <div>
            <h1 className="text-xl lg:text-2xl font-black text-white text-outfit tracking-tighter uppercase leading-none">
              SentFusion<span className="text-cyan-500">Net</span>
            </h1>
            <div className="flex items-center gap-2 mt-1">
              <Radio className="w-3 h-3 text-cyan-500 animate-pulse" />
              <span className="text-[9px] lg:text-[10px] font-bold text-slate-500 uppercase tracking-widest">Live Node: {data.market}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-3 w-full lg:w-auto">
          <button 
            onClick={() => window.print()} 
            className="flex-1 lg:flex-none flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-black font-black text-[10px] tracking-widest uppercase transition-all shadow-lg shadow-cyan-900/20"
          >
            <Layers className="w-4 h-4" /> Export Report
          </button>
          <div className="flex-1 lg:flex-none bg-slate-900/50 backdrop-blur-xl border border-slate-800 rounded-xl px-4 py-2 flex items-center justify-between lg:justify-start gap-3">
             <Globe className="w-4 h-4 text-slate-500" />
             <select 
               value={market} 
               onChange={(e) => setMarket(e.target.value)}
               className="bg-transparent text-slate-300 text-xs font-bold focus:outline-none cursor-pointer appearance-none"
             >
               <option value="us">US MARKET</option>
               <option value="india">INDIA MARKET</option>
             </select>
          </div>
          <button onClick={() => fetchForecast(true)} className="p-2.5 rounded-xl bg-slate-900/50 border border-slate-800 hover:bg-slate-800 transition-all group">
            <RefreshCcw className={`w-4 h-4 text-slate-400 group-hover:text-white ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </header>

      {/* MAIN CONTENT */}
      <main className="max-w-7xl mx-auto px-4 lg:px-8 mt-8 lg:mt-10 grid grid-cols-1 lg:grid-cols-12 gap-6 lg:gap-8">
        
        {/* HERO CARD */}
        <div className="col-span-1 lg:col-span-8 flex flex-col gap-6 lg:gap-8">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className={`glass-panel rounded-[1.5rem] lg:rounded-[2.5rem] p-6 lg:p-10 transition-all duration-500 ${isBullish ? 'bullish-glow' : 'bearish-glow'}`}
          >
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 mb-8">
               <div className="flex items-center gap-2 px-3 py-1 rounded-lg bg-slate-900/80 border border-slate-800 text-[9px] font-black text-slate-400 uppercase tracking-widest">
                 <Layers className="w-3.5 h-3.5" /> Neural Projection
               </div>
               <div className={`flex items-center gap-2 px-4 py-1.5 rounded-full text-[10px] font-black tracking-widest uppercase ${isBullish ? 'bg-cyan-500 text-black' : 'bg-rose-500 text-white'}`}>
                  {isBullish ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                  {data.signal}
               </div>
            </div>

            <div className="mb-10 text-center sm:text-left">
              <h2 className="text-4xl sm:text-6xl lg:text-7xl font-black text-white text-outfit tracking-tighter mb-4 lg:mb-2 leading-tight overflow-hidden text-ellipsis">
                ${data.predicted_price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
              </h2>
              <div className="flex flex-wrap items-center justify-center sm:justify-start gap-4">
                <span className={`text-2xl lg:text-3xl font-bold text-outfit ${isBullish ? 'text-cyan-400' : 'text-rose-400'}`}>
                  {data.predicted_change_pct > 0 ? '+' : ''}{data.predicted_change_pct.toFixed(4)}%
                </span>
                <div className="hidden sm:block h-4 w-[1px] bg-slate-800" />
                <div className="flex flex-col items-center sm:items-start">
                  <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest">Current Market</span>
                  <span className="text-xs lg:text-sm font-bold text-slate-300">${data.last_close_price.toLocaleString()}</span>
                </div>
              </div>
            </div>

            {/* NEURAL INSIGHT BOX */}
            <div className="bg-white/[0.03] border border-white/[0.05] rounded-2xl p-4 lg:p-6 mb-8 lg:mb-10">
              <div className="flex items-center gap-2 mb-2 text-cyan-500">
                <Zap className="w-4 h-4 fill-cyan-500" />
                <span className="text-[10px] font-black uppercase tracking-widest text-outfit">Neural Rationale</span>
              </div>
              <p className="text-slate-400 text-[11px] lg:text-xs leading-relaxed font-medium">
                The deep neural network has detected a <span className="text-white">{isBullish ? 'strong accumulation' : 'distributive'}</span> pattern. 
                The most influential driver for today's {data.signal} signal is <span className="text-cyan-400 font-bold">{Object.keys(data.feature_importance)[0].replace('_lag1','')}</span>.
              </p>
            </div>

            <div className="h-48 lg:h-64 -mx-6 lg:-mx-10 border-t border-slate-800/50">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={data.chart_data}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={isBullish ? "#22d3ee" : "#f43f5e"} stopOpacity={0.2}/>
                      <stop offset="95%" stopColor={isBullish ? "#22d3ee" : "#f43f5e"} stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <Area type="monotone" dataKey="price" stroke={isBullish ? "#22d3ee" : "#f43f5e"} strokeWidth={3} fill="url(#colorPrice)" />
                  <Tooltip contentStyle={{ background: '#0f172a', border: 'none', borderRadius: '12px', fontSize: '10px' }} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* LIVE NEWS STREAM */}
            <div className="mt-8 pt-8 border-t border-slate-800/50">
              <div className="flex items-center gap-2 mb-6 text-slate-500">
                <Globe className="w-3.5 h-3.5" />
                <span className="text-[9px] font-black uppercase tracking-widest">Neural News Stream</span>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {data.news && data.news.map((item, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-white/[0.02] border border-white/[0.03] hover:bg-white/[0.04] transition-all">
                    <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${item.score > 0 ? 'bg-cyan-500 shadow-[0_0_8px_rgba(6,182,212,0.5)]' : item.score < 0 ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.5)]' : 'bg-slate-600'}`} />
                    <div className="overflow-hidden">
                      <p className="text-[10px] text-slate-300 font-medium leading-relaxed line-clamp-2">{item.title}</p>
                      <p className="text-[8px] text-slate-500 mt-1 font-bold uppercase">{item.provider}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        </div>

        {/* SIDEBAR */}
        <div className="col-span-1 lg:col-span-4 flex flex-col gap-6">
          
          {/* NEURAL POWER DISTRIBUTION */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel rounded-[1.5rem] lg:rounded-[2rem] p-6 lg:p-8"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Neural Power Distribution</h3>
              <div className="px-2 py-0.5 rounded-full bg-cyan-500/10 border border-cyan-500/20 text-[8px] font-black text-cyan-400">XAI ENABLED</div>
            </div>
            <div className="flex h-3 w-full rounded-full overflow-hidden bg-slate-800">
               <motion.div initial={{ width: 0 }} animate={{ width: `${data.impact_distribution.Technical * 100}%` }} className="h-full bg-cyan-500" />
               <motion.div initial={{ width: 0 }} animate={{ width: `${data.impact_distribution.Macro * 100}%` }} className="h-full bg-indigo-500" />
               <motion.div initial={{ width: 0 }} animate={{ width: `${data.impact_distribution.Sentiment * 100}%` }} className="h-full bg-amber-500" />
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center sm:text-left">
               <div className="flex flex-col">
                 <span className="text-[8px] font-bold text-cyan-500 uppercase">Tech</span>
                 <span className="text-xs font-black text-white">{(data.impact_distribution.Technical * 100).toFixed(0)}%</span>
               </div>
               <div className="flex flex-col border-x border-white/5">
                 <span className="text-[8px] font-bold text-indigo-500 uppercase">Macro</span>
                 <span className="text-xs font-black text-white">{(data.impact_distribution.Macro * 100).toFixed(0)}%</span>
               </div>
               <div className="flex flex-col">
                 <span className="text-[8px] font-bold text-amber-500 uppercase">Sent</span>
                 <span className="text-xs font-black text-white">{(data.impact_distribution.Sentiment * 100).toFixed(0)}%</span>
               </div>
            </div>
          </motion.div>

          {/* NEURAL TRUST VERIFICATION */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel rounded-[1.5rem] lg:rounded-[2rem] p-6 lg:p-8 border-cyan-500/20 bg-gradient-to-br from-slate-900/50 to-cyan-900/10"
          >
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-[10px] font-black text-cyan-400 uppercase tracking-widest">Neural Trust Score</h3>
              <Zap className="w-4 h-4 text-cyan-400 shadow-glow" />
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="p-4 rounded-2xl bg-black/40 border border-white/5">
                <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">Accuracy</p>
                <p className="text-xl lg:text-2xl font-black text-white">{backtest.accuracy}%</p>
              </div>
              <div className="p-4 rounded-2xl bg-black/40 border border-white/5 text-center">
                <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">Win Rate</p>
                <p className="text-xl lg:text-2xl font-black text-white">{backtest.win_rate}</p>
              </div>
            </div>
          </motion.div>

          {/* NEURAL PULSE (Indicators) */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel rounded-[1.5rem] lg:rounded-[2rem] p-6 lg:p-8"
          >
            <div className="flex items-center justify-between mb-6 text-slate-300">
              <h3 className="text-[10px] font-black uppercase tracking-widest">Neural Pulse</h3>
              <Activity className="w-4 h-4" />
            </div>
            <div className="space-y-4">
              {Object.entries(data.indicators).map(([key, val]) => (
                <div key={key} className="flex justify-between items-center group">
                  <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest group-hover:text-cyan-400 transition-colors">{key}</span>
                  <span className="text-sm font-black text-white">{val.toFixed(2)}</span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* DECISION DRIVERS */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="glass-panel rounded-[1.5rem] lg:rounded-[2rem] p-6 lg:p-8"
          >
            <div className="flex items-center justify-between mb-8">
              <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Decision Drivers</h3>
              <PieChart className="w-4 h-4 text-slate-500" />
            </div>
            <div className="space-y-5 h-64 lg:h-auto lg:max-h-[400px] overflow-y-auto pr-2 custom-scroll">
              {Object.entries(data.feature_importance).map(([key, val]) => (
                <div key={key}>
                  <div className="flex justify-between text-[9px] font-bold uppercase tracking-widest mb-1.5">
                    <span className="text-slate-400">{key.replace('_lag1', '')}</span>
                    <span className="text-cyan-400">{(val * 100).toFixed(1)}%</span>
                  </div>
                  <div className="h-1 w-full bg-slate-800 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }}
                      animate={{ width: `${val * 100}%` }}
                      className="h-full bg-cyan-500"
                    />
                  </div>
                </div>
              ))}
            </div>
          </motion.div>
        </div>
      </main>
      
      <footer className="mt-12 px-8 text-center lg:text-left flex flex-col lg:flex-row justify-between items-center gap-4 opacity-20 pb-10">
        <p className="text-[8px] font-black tracking-[0.4em] uppercase text-outfit">SentFusionNet AI v3.5</p>
        <p className="text-[8px] font-black tracking-[0.4em] uppercase text-outfit">Verified Academic Engine</p>
      </footer>

      {/* PRINT-ONLY DISCLAIMER */}
      <div className="hidden print:block border-t-2 border-black mt-10 pt-6 text-[10px] text-slate-600">
        <p className="font-bold mb-2 uppercase tracking-widest">Neural Analysis Disclaimer</p>
        <p className="leading-relaxed">
          This report was generated by the SentFusionNet Hybrid Neural Architecture. Calculations are based on real-time NLP sentiment extraction fused with 
          technical and macro indicators. Historical accuracy (73.68%) is not a guarantee of future results. For academic demonstration purposes only.
        </p>
        <p className="mt-4 font-black">Generated on: {new Date().toLocaleString()} | Node: {data.market}</p>
      </div>
    </div>
  );
};

export default Dashboard;
