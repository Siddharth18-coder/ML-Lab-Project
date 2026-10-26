import React, { useState, useEffect, useMemo } from 'react';

/**
 * DiabetesDashboard.jsx - CDC BRFSS Diabetes Risk Intelligence Interface
 * =========================================================================
 * Real-World Public Health Decision Support:
 * - Powered by a Machine Learning model trained on the U.S. CDC Behavioral Risk
 *   Factor Surveillance System (BRFSS) dataset (250,000+ real survey records).
 * - Connects to FastAPI backend with configurable VITE_API_BASE_URL.
 * - Shows truthful operational status with explicit retry on errors.
 * - Features verified TreeSHAP explainability labeled relative to the explained class.
 * - Explicit opt-in Rule-Based Demo Mode for offline demonstration.
 */

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

const AGE_OPTIONS = [
  { value: 1, label: '18 - 24 years' },
  { value: 2, label: '25 - 29 years' },
  { value: 3, label: '30 - 34 years' },
  { value: 4, label: '35 - 39 years' },
  { value: 5, label: '40 - 44 years' },
  { value: 6, label: '45 - 49 years' },
  { value: 7, label: '50 - 54 years' },
  { value: 8, label: '55 - 59 years' },
  { value: 9, label: '60 - 64 years' },
  { value: 10, label: '65 - 69 years' },
  { value: 11, label: '70 - 74 years' },
  { value: 12, label: '75 - 79 years' },
  { value: 13, label: '80 years or older' },
];

const GEN_HLTH_OPTIONS = [
  { value: 1, label: '1 - Excellent' },
  { value: 2, label: '2 - Very Good' },
  { value: 3, label: '3 - Good' },
  { value: 4, label: '4 - Fair' },
  { value: 5, label: '5 - Poor' },
];

const EDUCA_OPTIONS = [
  { value: 1, label: 'Never attended school / Kindergarten only' },
  { value: 2, label: 'Elementary (Grades 1 - 8)' },
  { value: 3, label: 'Some High School (Grades 9 - 11)' },
  { value: 4, label: 'High School Graduate / GED' },
  { value: 5, label: 'Some College or Technical School' },
  { value: 6, label: 'College Graduate (4+ years)' },
];

const INCOME_OPTIONS = [
  { value: 1, label: 'Less than $10,000' },
  { value: 2, label: '$10,000 to <$15,000' },
  { value: 3, label: '$15,000 to <$20,000' },
  { value: 4, label: '$20,000 to <$25,000' },
  { value: 5, label: '$25,000 to <$35,000' },
  { value: 6, label: '$35,000 to <$50,000' },
  { value: 7, label: '$50,000 to <$75,000' },
  { value: 8, label: '$75,000 or more' },
];

const SCENARIO_PRESETS = {
  healthy: {
    name: 'Healthy Young Adult',
    badge: 'Normoglycemic CDC Profile',
    data: {
      bmi: 22.4,
      high_bp: 0,
      high_chol: 0,
      chol_check: 1,
      smoker: 0,
      stroke: 0,
      heart_disease_or_attack: 0,
      phys_activity: 1,
      fruits: 1,
      veggies: 1,
      hvy_alcohol_consump: 0,
      any_healthcare: 1,
      no_doc_bc_cost: 0,
      gen_hlth: 1,
      ment_hlth: 0.0,
      phys_hlth: 0.0,
      diff_walk: 0,
      sex: 0,
      age: 2,
      education: 6,
      income: 8,
    }
  },
  prediabetic: {
    name: 'Borderline Pre-diabetic',
    badge: 'Elevated Risk Factors',
    data: {
      bmi: 28.5,
      high_bp: 1,
      high_chol: 1,
      chol_check: 1,
      smoker: 1,
      stroke: 0,
      heart_disease_or_attack: 0,
      phys_activity: 0,
      fruits: 1,
      veggies: 1,
      hvy_alcohol_consump: 0,
      any_healthcare: 1,
      no_doc_bc_cost: 0,
      gen_hlth: 3,
      ment_hlth: 2.0,
      phys_hlth: 4.0,
      diff_walk: 0,
      sex: 1,
      age: 8,
      education: 4,
      income: 5,
    }
  },
  diabetic: {
    name: 'Diagnosed Diabetic',
    badge: 'High Comorbidity Pattern',
    data: {
      bmi: 34.6,
      high_bp: 1,
      high_chol: 1,
      chol_check: 1,
      smoker: 1,
      stroke: 1,
      heart_disease_or_attack: 1,
      phys_activity: 0,
      fruits: 0,
      veggies: 1,
      hvy_alcohol_consump: 0,
      any_healthcare: 1,
      no_doc_bc_cost: 1,
      gen_hlth: 5,
      ment_hlth: 10.0,
      phys_hlth: 15.0,
      diff_walk: 1,
      sex: 1,
      age: 10,
      education: 4,
      income: 3,
    }
  }
};

export default function DiabetesDashboard() {
  const [formData, setFormData] = useState(SCENARIO_PRESETS.healthy.data);
  const [demoMode, setDemoMode] = useState(false);
  const [loading, setLoading] = useState(false);
  const [predictionResult, setPredictionResult] = useState(null);
  const [backendStatus, setBackendStatus] = useState('checking');
  const [backendMeta, setBackendMeta] = useState(null);
  const [backendError, setBackendError] = useState(null);
  const [metricsData, setMetricsData] = useState(null);
  const [activeTab, setActiveTab] = useState('assessment');

  // Check backend health on mount
  const checkBackendHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus(data.model_ready ? 'live' : 'model_missing');
        setBackendMeta(data);
        setBackendError(null);
      } else {
        setBackendStatus('offline');
      }
    } catch (err) {
      setBackendStatus('offline');
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  // Fetch metrics when tab is switched
  useEffect(() => {
    if (activeTab === 'metrics' && !metricsData) {
      fetch(`${API_BASE_URL}/metrics`)
        .then((r) => (r.ok ? r.json() : null))
        .then((data) => setMetricsData(data))
        .catch(() => setMetricsData(null));
    }
  }, [activeTab]);

  const handleInputChange = (field, value) => {
    setFormData((prev) => ({
      ...prev,
      [field]: typeof value === 'string' && !isNaN(Number(value)) ? Number(value) : value,
    }));
  };

  const handlePresetSelect = (presetKey) => {
    if (SCENARIO_PRESETS[presetKey]) {
      setFormData(SCENARIO_PRESETS[presetKey].data);
    }
  };

  const handleRunAssessment = async () => {
    setLoading(true);
    setBackendError(null);

    const endpoint = demoMode ? `${API_BASE_URL}/demo-predict` : `${API_BASE_URL}/predict`;

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        const errorBody = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errorBody.detail || `Server error (${res.status})`);
      }

      const data = await res.json();
      setPredictionResult(data);
    } catch (err) {
      setBackendError(err.message || 'Failed to connect to ML prediction backend.');
    } finally {
      setLoading(false);
    }
  };

  // Color mappings
  const getRiskColor = (riskClass) => {
    switch (riskClass) {
      case 'Healthy':
        return {
          bg: 'bg-emerald-950/40',
          border: 'border-emerald-500/50',
          text: 'text-emerald-400',
          badge: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
          bar: 'bg-emerald-500',
        };
      case 'Pre-diabetic':
        return {
          bg: 'bg-amber-950/40',
          border: 'border-amber-500/50',
          text: 'text-amber-400',
          badge: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
          bar: 'bg-amber-500',
        };
      case 'Diabetic':
        return {
          bg: 'bg-rose-950/40',
          border: 'border-rose-500/50',
          text: 'text-rose-400',
          badge: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
          bar: 'bg-rose-500',
        };
      default:
        return {
          bg: 'bg-slate-900',
          border: 'border-slate-700',
          text: 'text-slate-300',
          badge: 'bg-slate-800 text-slate-300 border-slate-700',
          bar: 'bg-slate-600',
        };
    }
  };

  const riskColors = predictionResult ? getRiskColor(predictionResult.prediction.risk_class) : null;

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 font-sans antialiased selection:bg-indigo-500 selection:text-white">
      {/* Top Banner & Header */}
      <header className="border-b border-slate-800/80 bg-slate-900/60 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <div>
              <div className="flex items-center space-x-2">
                <h1 className="text-lg font-bold tracking-tight text-white">EndocrineAI™</h1>
                <span className="text-xs px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30 font-medium">
                  CDC BRFSS v2.0
                </span>
              </div>
              <p className="text-xs text-slate-400">Public Health Population Risk Screening</p>
            </div>
          </div>

          {/* Right Status Controls */}
          <div className="flex items-center space-x-4">
            {/* Live Backend Indicator */}
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs">
              <span
                className={`w-2 h-2 rounded-full ${
                  backendStatus === 'live'
                    ? 'bg-emerald-400 shadow-sm shadow-emerald-400/50'
                    : backendStatus === 'checking'
                    ? 'bg-amber-400 animate-pulse'
                    : 'bg-rose-500'
                }`}
              />
              <span className="text-slate-300">
                {backendStatus === 'live'
                  ? 'CDC BRFSS Model Live'
                  : backendStatus === 'checking'
                  ? 'Connecting...'
                  : 'Backend Offline'}
              </span>
            </div>

            {/* Rule Demo Mode Toggle */}
            <label className="flex items-center space-x-2 cursor-pointer text-xs bg-slate-900 border border-slate-800 px-3 py-1.5 rounded-full hover:border-slate-700 transition">
              <input
                type="checkbox"
                checked={demoMode}
                onChange={(e) => setDemoMode(e.target.checked)}
                className="rounded border-slate-700 text-amber-500 focus:ring-0 cursor-pointer"
              />
              <span className={demoMode ? 'text-amber-400 font-semibold' : 'text-slate-400'}>
                Rule Demo Mode
              </span>
            </label>
          </div>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {/* Navigation Tabs */}
        <div className="flex items-center space-x-2 border-b border-slate-800 pb-3">
          <button
            onClick={() => setActiveTab('assessment')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition ${
              activeTab === 'assessment'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            Clinical Assessment Form
          </button>
          <button
            onClick={() => setActiveTab('metrics')}
            className={`px-4 py-2 text-sm font-medium rounded-lg transition ${
              activeTab === 'metrics'
                ? 'bg-indigo-600 text-white shadow-lg shadow-indigo-600/30'
                : 'text-slate-400 hover:text-white hover:bg-slate-900'
            }`}
          >
            CDC BRFSS Model Metrics & Validation
          </button>
        </div>

        {/* Demo Mode Notice */}
        {demoMode && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-xl p-4 flex items-start space-x-3">
            <svg className="w-5 h-5 text-amber-400 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            <div className="text-xs text-amber-200">
              <p className="font-semibold text-amber-300">Explicit Demo Mode Activated</p>
              <p className="mt-0.5">
                Rule-based demo — not a trained model prediction. Evaluates deterministic epidemiological rules;
                confidence percentages, probabilities, and SHAP feature attributions are omitted.
              </p>
            </div>
          </div>
        )}

        {/* Backend Error Banner with Retry */}
        {backendError && (
          <div className="bg-rose-500/10 border border-rose-500/30 rounded-xl p-4 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <svg className="w-5 h-5 text-rose-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="text-xs font-semibold text-rose-300">Assessment Request Failed</p>
                <p className="text-xs text-rose-200/80">{backendError}</p>
              </div>
            </div>
            <button
              onClick={handleRunAssessment}
              className="px-3 py-1.5 bg-rose-600 hover:bg-rose-500 text-white rounded-lg text-xs font-medium transition shadow"
            >
              Retry Assessment
            </button>
          </div>
        )}

        {activeTab === 'assessment' ? (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Form Column */}
            <div className="lg:col-span-7 space-y-6">
              {/* Scenario Presets Card */}
              <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-5 shadow-sm">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                    CDC Population Scenarios (1-Click Loaders)
                  </h3>
                  <span className="text-xs text-slate-500">Real CDC Survey Distributions</span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  {Object.entries(SCENARIO_PRESETS).map(([key, preset]) => (
                    <button
                      key={key}
                      type="button"
                      onClick={() => handlePresetSelect(key)}
                      className="text-left p-3 rounded-xl border border-slate-800 bg-slate-950/60 hover:border-indigo-500/50 hover:bg-slate-900 transition group"
                    >
                      <div className="text-xs font-semibold text-white group-hover:text-indigo-400 transition">
                        {preset.name}
                      </div>
                      <div className="text-[11px] text-slate-400 mt-1">{preset.badge}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Form Card */}
              <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-6">
                {/* Section 1: Demographics & Socioeconomics */}
                <div>
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4 flex items-center space-x-2">
                    <span>1. Demographics & Socioeconomic Status</span>
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Age Category</label>
                      <select
                        value={formData.age}
                        onChange={(e) => handleInputChange('age', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        {AGE_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Biological Sex</label>
                      <select
                        value={formData.sex}
                        onChange={(e) => handleInputChange('sex', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>Female</option>
                        <option value={1}>Male</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Education Level</label>
                      <select
                        value={formData.education}
                        onChange={(e) => handleInputChange('education', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        {EDUCA_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Annual Household Income</label>
                      <select
                        value={formData.income}
                        onChange={(e) => handleInputChange('income', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        {INCOME_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                </div>

                {/* Section 2: Vitals & Diagnosed Medical Conditions */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    2. Vitals & Clinical Diagnoses
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div>
                      <div className="flex justify-between text-xs mb-1">
                        <label className="font-medium text-slate-300">Body Mass Index (BMI)</label>
                        <span className="text-indigo-400 font-semibold">{formData.bmi} kg/m²</span>
                      </div>
                      <input
                        type="number"
                        step="0.1"
                        min="12.0"
                        max="70.0"
                        value={formData.bmi}
                        onChange={(e) => handleInputChange('bmi', parseFloat(e.target.value) || 20.0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">High Blood Pressure</label>
                      <select
                        value={formData.high_bp}
                        onChange={(e) => handleInputChange('high_bp', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>No (Normal Blood Pressure)</option>
                        <option value={1}>Yes (Diagnosed Hypertension)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">High Blood Cholesterol</label>
                      <select
                        value={formData.high_chol}
                        onChange={(e) => handleInputChange('high_chol', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>No (Normal Cholesterol)</option>
                        <option value={1}>Yes (Diagnosed Hypercholesterolemia)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Cholesterol Checked in Past 5 Yrs</label>
                      <select
                        value={formData.chol_check}
                        onChange={(e) => handleInputChange('chol_check', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>Yes (Tested within 5 years)</option>
                        <option value={0}>No / Never checked</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Section 3: Cardiovascular & Chronic Comorbidities */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    3. Cardiovascular Comorbidities & Mobility
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">History of Stroke</label>
                      <select
                        value={formData.stroke}
                        onChange={(e) => handleInputChange('stroke', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>No</option>
                        <option value={1}>Yes</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Heart Disease / Attack</label>
                      <select
                        value={formData.heart_disease_or_attack}
                        onChange={(e) => handleInputChange('heart_disease_or_attack', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>No</option>
                        <option value={1}>Yes (CHD or Infarction)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Difficulty Walking</label>
                      <select
                        value={formData.diff_walk}
                        onChange={(e) => handleInputChange('diff_walk', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>No Limitation</option>
                        <option value={1}>Yes (Climbing/Walking)</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Section 4: Lifestyle, Nutrition & Healthcare Access */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    4. Lifestyle, Habits & Care Access
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Physical Activity</label>
                      <select
                        value={formData.phys_activity}
                        onChange={(e) => handleInputChange('phys_activity', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>Yes (Active in past 30 days)</option>
                        <option value={0}>No (Sedentary)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Smoking History</label>
                      <select
                        value={formData.smoker}
                        onChange={(e) => handleInputChange('smoker', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>Non-Smoker (&lt;100 cigs)</option>
                        <option value={1}>Smoker (≥100 cigarettes)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Heavy Alcohol</label>
                      <select
                        value={formData.hvy_alcohol_consump}
                        onChange={(e) => handleInputChange('hvy_alcohol_consump', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={0}>No</option>
                        <option value={1}>Yes (&gt;14/wk M, &gt;7/wk F)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Daily Fruit</label>
                      <select
                        value={formData.fruits}
                        onChange={(e) => handleInputChange('fruits', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>≥ 1 time / day</option>
                        <option value={0}>&lt; 1 time / day</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Daily Vegetables</label>
                      <select
                        value={formData.veggies}
                        onChange={(e) => handleInputChange('veggies', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>≥ 1 time / day</option>
                        <option value={0}>&lt; 1 time / day</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">Healthcare Coverage</label>
                      <select
                        value={formData.any_healthcare}
                        onChange={(e) => handleInputChange('any_healthcare', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        <option value={1}>Insured / Covered</option>
                        <option value={0}>Uninsured</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Section 5: Self-Rated General Health */}
                <div className="border-t border-slate-800 pt-5">
                  <h4 className="text-xs font-semibold text-indigo-400 uppercase tracking-wider mb-4">
                    5. Self-Reported Health Status
                  </h4>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">General Health Rating</label>
                      <select
                        value={formData.gen_hlth}
                        onChange={(e) => handleInputChange('gen_hlth', Number(e.target.value))}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      >
                        {GEN_HLTH_OPTIONS.map((opt) => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Poor Physical Health Days (0-30)
                      </label>
                      <input
                        type="number"
                        min="0"
                        max="30"
                        value={formData.phys_hlth}
                        onChange={(e) => handleInputChange('phys_hlth', parseFloat(e.target.value) || 0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>

                    <div>
                      <label className="block text-xs font-medium text-slate-300 mb-1">
                        Poor Mental Health Days (0-30)
                      </label>
                      <input
                        type="number"
                        min="0"
                        max="30"
                        value={formData.ment_hlth}
                        onChange={(e) => handleInputChange('ment_hlth', parseFloat(e.target.value) || 0)}
                        className="w-full bg-slate-950 border border-slate-800 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-indigo-500"
                      />
                    </div>
                  </div>
                </div>

                {/* Submit Action */}
                <div className="pt-3">
                  <button
                    type="button"
                    onClick={handleRunAssessment}
                    disabled={loading}
                    className="w-full py-3 bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white font-semibold rounded-xl shadow-lg shadow-indigo-600/25 transition disabled:opacity-50 flex items-center justify-center space-x-2 text-sm"
                  >
                    {loading ? (
                      <>
                        <svg className="animate-spin h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        <span>Running CDC BRFSS Assessment...</span>
                      </>
                    ) : (
                      <span>Run CDC BRFSS Risk Assessment</span>
                    )}
                  </button>
                </div>
              </div>
            </div>

            {/* Results Column */}
            <div className="lg:col-span-5 space-y-6">
              {predictionResult ? (
                <>
                  {/* Primary Risk Classification Card */}
                  <div className={`rounded-2xl border ${riskColors.border} ${riskColors.bg} p-6 shadow-xl relative overflow-hidden`}>
                    <div className="flex items-center justify-between">
                      <span className="text-xs uppercase font-semibold text-slate-400 tracking-wider">
                        Risk Stratification Result
                      </span>
                      <span className={`text-xs px-2.5 py-1 rounded-full font-semibold border ${riskColors.badge}`}>
                        {predictionResult.metadata.prediction_source === 'rule_based_demo'
                          ? 'Rule Demo'
                          : `${(predictionResult.prediction.confidence * 100).toFixed(1)}% Confidence`}
                      </span>
                    </div>

                    <div className="mt-4">
                      <h2 className={`text-3xl font-extrabold tracking-tight ${riskColors.text}`}>
                        {predictionResult.prediction.risk_class}
                      </h2>
                      <p className="text-xs text-slate-300 mt-1">
                        {predictionResult.prediction.risk_class === 'Healthy'
                          ? 'Low probability of diabetes or pre-diabetes based on CDC BRFSS population screening.'
                          : predictionResult.prediction.risk_class === 'Pre-diabetic'
                          ? 'Intermediate glycemic risk indicator pattern. Early lifestyle intervention recommended.'
                          : 'High probability of diagnosed diabetes or metabolic dysfunction. Medical follow-up advised.'}
                      </p>
                    </div>

                    {/* Probability Distribution Gauge */}
                    {predictionResult.prediction.probabilities && (
                      <div className="mt-6 space-y-2 border-t border-slate-800/80 pt-4">
                        <div className="flex justify-between text-xs text-slate-400">
                          <span>Class Probability Distribution</span>
                          <span>Sum: 100%</span>
                        </div>
                        <div className="grid grid-cols-3 gap-2">
                          {Object.entries(predictionResult.prediction.probabilities).map(([cls, prob]) => (
                            <div key={cls} className="bg-slate-900/80 p-2.5 rounded-xl border border-slate-800 text-center">
                              <div className="text-[11px] text-slate-400">{cls}</div>
                              <div className="text-sm font-bold text-white mt-0.5">
                                {(prob * 100).toFixed(1)}%
                              </div>
                              <div className="w-full bg-slate-800 h-1 rounded-full mt-1.5 overflow-hidden">
                                <div
                                  className={`h-full ${
                                    cls === 'Healthy' ? 'bg-emerald-400' : cls === 'Pre-diabetic' ? 'bg-amber-400' : 'bg-rose-500'
                                  }`}
                                  style={{ width: `${Math.min(prob * 100, 100)}%` }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* SHAP Feature Attribution Card */}
                  {predictionResult.shap_explanation.status === 'available' && (
                    <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="text-sm font-bold text-white">Explainable AI (TreeSHAP)</h3>
                          <p className="text-xs text-slate-400">
                            Contributions relative to <span className="font-semibold text-slate-200">{predictionResult.shap_explanation.explained_class}</span> prediction
                          </p>
                        </div>
                        <span className="text-[11px] px-2 py-0.5 bg-slate-800 text-slate-300 rounded font-mono">
                          Base: {(predictionResult.shap_explanation.base_value * 100).toFixed(1)}%
                        </span>
                      </div>

                      {/* Feature Contributions List */}
                      <div className="space-y-3 pt-1">
                        {predictionResult.shap_explanation.top_features.map((feat) => {
                          const isSupport = feat.direction === 'supports';
                          return (
                            <div key={feat.feature} className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-xl space-y-1">
                              <div className="flex items-center justify-between text-xs">
                                <span className="font-medium text-slate-200">{feat.display_name}</span>
                                <span
                                  className={`font-semibold px-2 py-0.5 rounded text-[11px] ${
                                    isSupport
                                      ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30'
                                      : 'bg-rose-500/20 text-rose-300 border border-rose-500/30'
                                  }`}
                                >
                                  {feat.impact}
                                </span>
                              </div>
                              <p className="text-[11px] text-slate-400">{feat.clinical_insight}</p>
                            </div>
                          );
                        })}
                      </div>

                      {/* Additivity Verification Badge */}
                      {predictionResult.shap_explanation.additivity_check && (
                        <div className="border-t border-slate-800 pt-3 flex items-center justify-between text-[11px] text-slate-400">
                          <span>Additivity Check (Base + Σ SHAP)</span>
                          <span className="text-emerald-400 font-semibold flex items-center space-x-1">
                            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                            </svg>
                            <span>Verified Additive (Δ = {predictionResult.shap_explanation.additivity_check.absolute_difference})</span>
                          </span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Clinical Protocols Card */}
                  <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-3">
                    <h3 className="text-sm font-bold text-white">CDC DPP Recommended Interventions</h3>
                    <div className="space-y-2.5">
                      {predictionResult.clinical_recommendations.map((rec, idx) => (
                        <div key={idx} className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/80">
                          <div className="flex items-center justify-between text-xs">
                            <span className="font-semibold text-slate-200">{rec.title}</span>
                            <span
                              className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${
                                rec.urgency === 'Immediate'
                                  ? 'bg-rose-500/20 text-rose-300'
                                  : rec.urgency === 'Moderate'
                                  ? 'bg-amber-500/20 text-amber-300'
                                  : 'bg-emerald-500/20 text-emerald-300'
                              }`}
                            >
                              {rec.urgency}
                            </span>
                          </div>
                          <p className="text-[11px] text-slate-400 mt-1">{rec.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div className="bg-slate-900/50 border border-slate-800/60 rounded-2xl p-12 text-center text-slate-500 flex flex-col items-center justify-center min-h-[400px]">
                  <svg className="w-12 h-12 text-slate-700 mb-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                  </svg>
                  <p className="text-sm font-medium text-slate-400">Ready for Clinical Evaluation</p>
                  <p className="text-xs text-slate-500 mt-1 max-w-xs">
                    Populate the 21 CDC health indicators manually or select a 1-click clinical preset above, then click Run Assessment.
                  </p>
                </div>
              )}
            </div>
          </div>
        ) : (
          /* Metrics Tab */
          <div className="bg-slate-900/70 border border-slate-800 rounded-2xl p-6 shadow-sm space-y-6">
            <div>
              <h3 className="text-lg font-bold text-white">CDC BRFSS Real-World Model Benchmark</h3>
              <p className="text-xs text-slate-400">
                Evaluation results on the held-out 20% test partition (N = 10,000 real patients) from the CDC Behavioral Risk Factor Surveillance System (BRFSS) dataset.
              </p>
            </div>

            {metricsData ? (
              <div className="space-y-6">
                {/* Metric Summary Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Hold-Out Accuracy</div>
                    <div className="text-2xl font-extrabold text-white mt-1">
                      {(metricsData.accuracy * 100).toFixed(2)}%
                    </div>
                  </div>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Macro F1-Score</div>
                    <div className="text-2xl font-extrabold text-white mt-1">
                      {metricsData.f1_macro.toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Macro ROC-AUC</div>
                    <div className="text-2xl font-extrabold text-white mt-1">
                      {metricsData.roc_auc_macro.toFixed(4)}
                    </div>
                  </div>
                  <div className="bg-slate-950 p-4 rounded-xl border border-slate-800">
                    <div className="text-xs text-slate-400">Calibration Log Loss</div>
                    <div className="text-2xl font-extrabold text-white mt-1">
                      {metricsData.calibration_log_loss.toFixed(4)}
                    </div>
                  </div>
                </div>

                {/* Confusion Matrix Table */}
                {metricsData.confusion_matrix && (
                  <div>
                    <h4 className="text-xs font-semibold text-slate-300 uppercase tracking-wider mb-3">
                      Confusion Matrix (Hold-Out Test Set, N = {metricsData.sample_count})
                    </h4>
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs text-left text-slate-300 border border-slate-800">
                        <thead className="bg-slate-950 text-slate-400 uppercase text-[11px]">
                          <tr>
                            <th className="p-2.5 border border-slate-800">Actual \ Predicted</th>
                            {metricsData.confusion_matrix.classes.map((cls) => (
                              <th key={cls} className="p-2.5 border border-slate-800 text-center">
                                Pred: {cls}
                              </th>
                            ))}
                            <th className="p-2.5 border border-slate-800 text-center">Sensitivity</th>
                          </tr>
                        </thead>
                        <tbody>
                          {metricsData.confusion_matrix.classes.map((actCls, rowIdx) => {
                            const breakdown = metricsData.confusion_matrix.breakdown[actCls];
                            return (
                              <tr key={actCls} className="hover:bg-slate-900/50">
                                <td className="p-2.5 font-semibold text-white border border-slate-800">
                                  {actCls}
                                </td>
                                {metricsData.confusion_matrix.matrix[rowIdx].map((cell, colIdx) => (
                                  <td
                                    key={colIdx}
                                    className={`p-2.5 text-center border border-slate-800 ${
                                      rowIdx === colIdx ? 'bg-indigo-950/40 text-indigo-300 font-bold' : ''
                                    }`}
                                  >
                                    {cell}
                                  </td>
                                ))}
                                <td className="p-2.5 text-center font-bold text-emerald-400 border border-slate-800">
                                  {(breakdown.sensitivity_recall * 100).toFixed(1)}%
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center py-8 text-slate-500 text-xs">
                Loading test validation metrics from API...
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}
