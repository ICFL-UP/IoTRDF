import React, { useState, useEffect, useRef } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { 
  faEye, faInfoCircle, faCheckCircle, faDatabase, faTools, 
  faChartLine, faPlayCircle, faRobot, faFlask, faVial, 
  faArrowLeft, faCog, faSearch, faMedal, faSpinner,
  faChartBar, faChartArea, faDownload
} from "@fortawesome/free-solid-svg-icons";

import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Line, Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const App = () => {
  const [currentStage, setCurrentStage] = useState("Start");
  const [logs, setLogs] = useState(["System initialized. Ready to begin..."]);
  const [policyLogs, setPolicyLogs] = useState([]);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [workAreaMessage, setWorkAreaMessage] = useState("Click 'Start Experiment' to begin.");
  const [datasets, setDatasets] = useState([]);
  const [selectedDataset, setSelectedDataset] = useState(null);
  const [preprocessingComplete, setPreprocessingComplete] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [policyTrained, setPolicyTrained] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [testingComplete, setTestingComplete] = useState(false);
  const [bestFeatures, setBestFeatures] = useState([]);
  const [testResults, setTestResults] = useState(null);
  const [preprocessingResults, setPreprocessingResults] = useState(null);
  const [featureMap, setFeatureMap] = useState({});
  const [featureLookupLogs, setFeatureLookupLogs] = useState([]);
  const [currentFeatureSet, setCurrentFeatureSet] = useState([]);
  const [bestPolicyLogs, setBestPolicyLogs] = useState([]);
  const bestPoliciesRef = useRef([]);
  const bestMetricsRef = useRef({ f1: 0, size: Infinity, features: [] }); 
  const [bestF1SoFar, setBestF1SoFar] = useState(0);
  const [bestFeatureCount, setBestFeatureCount] = useState(Infinity);
  const [testingLogs, setTestingLogs] = useState([]);
  const [showPerformancePopup, setShowPerformancePopup] = useState(false);
  const [showFeaturesPopup, setShowFeaturesPopup] = useState(false);
  const [visualizationData, setVisualizationData] = useState({
    showTransitions: false,
    accuracyHistory: [],
    featureChanges: [],
  });

  const [hyperparameters, setHyperparameters] = useState({
    epochs: 200,
    warmup_steps: 64,
    batch_size: 32,
    buffer_size: 10000,
    learning_rate: 0.00003,
    actor_learning_rate: 0.00003,
    alpha_learning_rate: 0.0003,
    gamma: 0.99,
    tau: 0.001,
    target_entropy_ratio: 0.98
  });

  const [showHyperparameterModal, setShowHyperparameterModal] = useState(false);
  const [featureLookup, setFeatureLookup] = useState([]);
  const [bestPolicies, setBestPolicies] = useState([]);
  
  const [selectedClassifier, setSelectedClassifier] = useState("");
  const classifiers = [
    { value: "DT", label: "Decision Tree (DT)" },
    { value: "LR", label: "Logistic Regression (LR)" },
    { value: "KNN", label: "K-Nearest Neighbors (KNN)" },
    { value: "RF", label: "Random Forest (RF)" },
    { value: "SVM", label: "Support Vector Machine (SVM)" }
  ];

  const [trainingMetrics, setTrainingMetrics] = useState({
    epochs: [],
    accuracy: [],
    f1Scores: [],
    featureCounts: [],
    rewards: []
  });

  const [showPerformanceGraph, setShowPerformanceGraph] = useState(false);
  const [showFeaturesGraph, setShowFeaturesGraph] = useState(false);

  const stages = [
    { 
      name: "Environment", 
      icon: <FontAwesomeIcon icon={faDatabase} />, 
      description: "Upload or select dataset" 
    },
    { 
      name: "Preprocessing", 
      icon: <FontAwesomeIcon icon={faTools} />, 
      description: "Prepare dataset for training" 
    },
    { 
      name: "Policy Training", 
      icon: <FontAwesomeIcon icon={faRobot} />, 
      description: "Train SAC policy model" 
    },
    { 
      name: "Policy Testing", 
      icon: <FontAwesomeIcon icon={faVial} />, 
      description: "Test trained policy" 
    },
  ];

  const stageMessages = {
    "Data": "Please provide your IoT dataset to begin the experiment",
    "Preprocessing": "Preparing dataset for analysis...",
    "Policy Training": policyTrained 
      ? "Policy training complete. Ready for testing."
      : "Configure and train the policy model",
    "Policy Testing": testingComplete
      ? "Policy testing complete. View results."
      : "Evaluate the trained policy model",
  };

  const colors = {
    background: "#f8f9fa",
    sidebar: "#ffffff",
    card: "#ffffff",
    primary: "#4285f4",
    success: "#34a853",
    warning: "#fbbc05",
    danger: "#ea4335",
    text: "#202124",
    textSecondary: "#5f6368",
    border: "#dadce0"
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        font: {
          size: 14,
          weight: 'bold'
        }
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: 'Epochs',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(0,0,0,0.05)',
        }
      },
      y: {
        title: {
          display: true,
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        grid: {
          color: 'rgba(0,0,0,0.05)',
        }
      },
    },
  };

  const accuracyChartData = {
  labels: trainingMetrics.epochs,
  datasets: [
    {
      label: 'Accuracy (%)',
      data: trainingMetrics.accuracy,
      borderColor: 'rgb(53, 162, 235)',
      backgroundColor: 'rgba(53, 162, 235, 0.1)',
      tension: 0.1,
      pointRadius: 2,
      pointHoverRadius: 5,
      fill: true, 
    }
  
  ],
};

  const accuracyChartOptions = {
  ...chartOptions,
  plugins: {
    ...chartOptions.plugins,
    title: {
      display: true,
      text: 'Training Performance Metrics',
      font: {
        size: 14,
        weight: 'bold'
      }
    },
  },
  scales: {
    ...chartOptions.scales,
    y: {
      ...chartOptions.scales.y,
      title: {
        display: true,
        text: 'Accuracy (%)',
        font: {
          size: 12,
          weight: 'bold'
        }
      },
      min: 0,
      max: 100, 
      ticks: {
        callback: function(value) {
          return value + '%'; 
        }
      }
    }
  },
};

  const featureChartData = {
    labels: trainingMetrics.epochs,
    datasets: [
      {
        label: 'Number of Selected Features',
        data: trainingMetrics.featureCounts,
        borderColor: 'rgb(255, 99, 132)',
        backgroundColor: 'rgba(255, 99, 132, 0.1)',
        tension: 0.1,
        pointRadius: 2,
        pointHoverRadius: 5,
        fill: true,
      }
    ],
  };

  const featureChartOptions = {
    ...chartOptions,
    plugins: {
      ...chartOptions.plugins,
      title: {
        display: true,
        text: 'Feature Selection Progress',
        font: {
          size: 14,
          weight: 'bold'
        }
      },
    },
    scales: {
      ...chartOptions.scales,
      y: {
        ...chartOptions.scales.y,
        title: {
          display: true,
          text: 'Number of Features',
          font: {
            size: 12,
            weight: 'bold'
          }
        },
        min: 0,
        ticks: {
          stepSize: 1
        }
      }
    },
  };

  const updateTrainingMetrics = (epoch, accuracy, f1Score, featureCount, reward) => {
    setTrainingMetrics(prev => ({
      epochs: [...prev.epochs, epoch],
      accuracy: [...prev.accuracy, accuracy * 100],
      f1Scores: [...prev.f1Scores, f1Score],
      featureCounts: [...prev.featureCounts, featureCount],
      rewards: [...prev.rewards, reward]
    }));
  };

  const resetTrainingMetrics = () => {
    setTrainingMetrics({
      epochs: [],
      accuracy: [],
      f1Scores: [],
      featureCounts: [],
      rewards: []
    });
  };

  useEffect(() => {
    fetchDatasets();
  }, []);

  const fetchDatasets = async () => {
    try {
      const response = await fetch("http://localhost:5000/datasets");
      if (!response.ok) throw new Error("Failed to load datasets");
      const data = await response.json();
      setDatasets(data.datasets || []);
      setLogs(prev => [...prev, "Available datasets loaded"]);
    } catch (error) {
      setLogs(prev => [...prev, `Failed to load datasets: ${error.message}`]);
    }
  };

  useEffect(() => {
    if (selectedDataset) {
      fetchFeatureLookup();
    }
  }, [selectedDataset]);

  useEffect(() => {
    if (selectedDataset && policyTrained) {
      fetchBestPolicies();
    }
  }, [selectedDataset, policyTrained]);

  const fetchFeatureLookup = async () => {
    try {
      const response = await fetch(`http://localhost:5000/feature_lookup?dataset_id=${selectedDataset}`);
      if (!response.ok) throw new Error("Feature lookup failed");
      const data = await response.json();
      setFeatureLookup(data.feature_map || []);
    } catch (error) {
      console.error("Error fetching feature lookup:", error);
      setLogs(prev => [...prev, `Feature lookup error: ${error.message}`]);
    }
  };

  const fetchBestPolicies = async () => {
    try {
      const response = await fetch(`http://localhost:5000/best_policies?dataset_id=${selectedDataset}`);
      if (!response.ok) throw new Error("Failed to load best policies");
      const data = await response.json();
      setBestPolicies(data.policies || []);
    } catch (error) {
      console.error("Error fetching best policies:", error);
      setLogs(prev => [...prev, `Best policies error: ${error.message}`]);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file && (file.type === "text/csv" || file.type.includes("excel"))) {
      setUploadedFile(file);
      setWorkAreaMessage("Dataset file selected. Click submit to continue.");
      setLogs(prev => [...prev, `Selected file: ${file.name}`]);
    } else {
      setUploadedFile(null);
      setWorkAreaMessage("Please select a valid CSV or Excel file.");
      setLogs(prev => [...prev, "Invalid file type"]);
    }
  };

  const handleSubmitFile = async () => {
    if (!uploadedFile && !selectedDataset) {
      setLogs(prev => [...prev, "Error: No dataset selected or uploaded"]);
      return;
    }

    try {
      setLogs(prev => [...prev, "Uploading dataset..."]);
      setIsProcessing(true);

      let resolvedId = Number(selectedDataset) || null;

      if (uploadedFile) {
        const formData = new FormData();
        formData.append("file", uploadedFile);

        const response = await fetch("http://localhost:5000/upload", { 
          method: "POST", 
          body: formData 
        });
        if (!response.ok) throw new Error("Upload failed");

        const data = await response.json();
        resolvedId = Number(data.dataset_id);
        setSelectedDataset(resolvedId);
        setLogs(prev => [...prev, `Dataset uploaded (ID: ${resolvedId})`]);
      } else {
        setLogs(prev => [...prev, `Using existing dataset (ID: ${resolvedId})`]);
      }

      setWorkAreaMessage("Starting preprocessing...");
      setCurrentStage("Preprocessing");

      await handlePreprocessing(resolvedId);
    } catch (error) {
      setLogs(prev => [...prev, `Dataset upload failed: ${error.message}`]);
      setIsProcessing(false);
    }
  };


  const handlePreprocessing = async (datasetId) => {
  try {
  
    const id = Number(datasetId ?? selectedDataset);
    if (!id) {
      setLogs(prev => [...prev, "Error: Missing dataset_id for preprocessing"]);
      return;
    }

    setPreprocessingComplete(false);
    setPreprocessingResults(null);

    setPolicyTrained(false);
    setPolicyLogs([]);
    setBestPolicies([]);
    setBestFeatures([]);
    setTestingComplete(false);
    setTestResults(null);
    setFeatureLookupLogs([]);
    setCurrentFeatureSet([]);
    setBestF1SoFar(0);
    setBestFeatureCount(Infinity);
    setSelectedClassifier("");
    resetTrainingMetrics();
    setShowPerformanceGraph(false);
    setShowFeaturesGraph(false);

    setLogs(prev => [...prev, "Starting dataset preprocessing..."]);
    setWorkAreaMessage("Preprocessing in progress...");
    setIsProcessing(true);

    const response = await fetch("http://localhost:5000/preprocess", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_id: id })
    });

    if (!response.ok) {
 
      let serverMsg = "";
      try {
        const errJson = await response.json();
        serverMsg = errJson?.error || "";
      } catch (_) {}
      throw new Error(serverMsg || `HTTP error! Status: ${response.status}`);
    }

    const data = await response.json();

    const getCount = (shapeStr) => parseInt((shapeStr || "").split(" ")[0], 10);
    const trainCount = getCount(data.X_train_shape);
    const valCount   = getCount(data.X_val_shape);
    const testCount  = getCount(data.X_test_shape);
    const totalCount = (trainCount || 0) + (valCount || 0) + (testCount || 0);
    const featureCount = (data.X_train_shape || "").split("×")[1]?.trim(); // e.g., "63 features"

    setPreprocessingComplete(true);
    setPreprocessingResults(data);

    setLogs(prev => [
      ...prev,
      "Preprocessing completed successfully",
      <div key="train-features">
        Train Dataset: {data.X_train_shape}
        <FontAwesomeIcon
          icon={faEye}
          onClick={() => openCSV("train_features")}
          style={{ marginLeft: "10px", cursor: "pointer" }}
          title="View Train Features"
        />
      </div>,
      <div key="val-features">
        Validation Dataset: {data.X_val_shape}
        <FontAwesomeIcon
          icon={faEye}
          onClick={() => openCSV("val_features")}
          style={{ marginLeft: "10px", cursor: "pointer" }}
          title="View Validation Features"
        />
      </div>,
      <div key="test-features">
        Test Dataset: {data.X_test_shape}
        <FontAwesomeIcon
          icon={faEye}
          onClick={() => openCSV("test_features")}
          style={{ marginLeft: "10px", cursor: "pointer" }}
          title="View Test Features"
        />
      </div>,
      <div key="summary">
        Dataset Summary:
        <div>Total Samples: {totalCount}</div>
        <div>Features per Sample: {featureCount || 'Unknown'}</div>
        <div>
          Split Ratio: {totalCount ? Math.round((trainCount / totalCount) * 100) : 0}% Train /
          {totalCount ? Math.round((valCount / totalCount) * 100) : 0}% Val /
          {totalCount ? Math.round((testCount / totalCount) * 100) : 0}% Test
        </div>
      </div>,
      "Ready for Policy Training"
    ]);

    setWorkAreaMessage("Preprocessing complete. Ready for policy training.");
  } catch (error) {
    setWorkAreaMessage("Preprocessing failed. Please try again.");
    setLogs(prev => [...prev, `Error: ${error.message}`]);
  } finally {
    setIsProcessing(false);
  }
};

  const openCSV = (datasetType) => {
    window.open(`http://localhost:5000/preprocessing_results/${selectedDataset}?download_csv=true&dataset_type=${datasetType}`);
  };

const handlePolicyTraining = () => {

  resetTrainingMetrics();
  setShowPerformanceGraph(false);
  setShowFeaturesGraph(false);
  
  setWorkAreaMessage("Training the policy model. Please wait...");
  setIsProcessing(true);
  setPolicyTrained(false);
  setVisualizationData(prev => ({
    ...prev,
    showTransitions: false,
    accuracyHistory: [],
    featureChanges: [],
  }));
  setPolicyLogs([]);
  setBestFeatures([]);
  setBestPolicies([]);
  bestPoliciesRef.current = [];
  setFeatureLookupLogs([]);
  setFeatureMap({});
  setCurrentFeatureSet([]);
  setBestF1SoFar(0);
  setBestFeatureCount(Infinity);

  if (!selectedDataset) {
    setLogs(prevLogs => [...prevLogs, "Error: No dataset selected."]);
    setIsProcessing(false);
    return;
  }

  if (!selectedClassifier) {
    setLogs(prevLogs => [...prevLogs, "Error: No classifier selected."]);
    setIsProcessing(false);
    return;
  }

  const classifierMap = {
    DT: "dt",
    LR: "logreg",
    KNN: "knn",
    RF: "rf",
    SVM: "svm",
  };
  const backendClassifier = classifierMap[selectedClassifier];
  if (!backendClassifier) {
    setLogs(prevLogs => [...prevLogs, "Error: Invalid classifier selected for backend."]);
    setIsProcessing(false);
    return;
  }

  const query = new URLSearchParams({
    dataset_id: selectedDataset,
    classifier: backendClassifier,
  });
  for (const key in hyperparameters) {
    if (Object.prototype.hasOwnProperty.call(hyperparameters, key)) {
      query.append(key, hyperparameters[key]);
    }
  }

  fetch(`http://localhost:5000/feature_lookup?${query}`)
    .then(res => res.json())
    .then(data => {
      if (data.status === "success") {
        setFeatureMap(data.feature_map || {});
      } else {
        setLogs(prev => [
          ...prev,
          <div style={{ color: "red" }}> Failed to load feature map.</div>,
        ]);
      }
    });

  let bestF1Run = 0;
  let bestSizeRun = Infinity;
  let bestSetRun = [];
  const EPS = 1e-4;

  let lastEpochSeen = 0;

  let trainingCompleted = false;
  const eventSource = new EventSource(`http://localhost:5000/train_policy_model?${query}`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (typeof data.epoch === "number") {
        lastEpochSeen = Math.max(lastEpochSeen, data.epoch);
      }
      console.log("Received data from backend:", data);

      if (data.error) {
        setLogs(prevLogs => [
          ...prevLogs,
          <div key={`error-${Date.now()}`} style={{ color: "red" }}>
            Error: {data.error} (Stage: {data.stage})
          </div>,
        ]);
        eventSource.close();
        setIsProcessing(false);
        return;
      }

      if (data.selected_features && Array.isArray(data.selected_features)) {
        const dedup = Array.from(new Set(data.selected_features));
        const mappedNames = dedup.map(idx => featureMap[idx] || `F${idx}`);
        const logEntry = `Epoch ${data.epoch} → Selected Set: {${dedup.join(", ")}} → Lookup: {${mappedNames.join(", ")}}`;
        setFeatureLookupLogs(prev => [...prev, logEntry]);
      }

      if (data.stage === "operation_decision") {
        const opProbs = data.operation_probs || {};
        const meanQ = data.mean_q_values || {};
        const maxQPotentials = data.max_q_potentials || {};

        const meanQRendered = Object.entries(meanQ).map(([key, value]) => (
          <div key={`meanq-${data.epoch}-${key}`} style={{ fontSize: "0.9em", color: "#1E88E5", paddingLeft: "8px" }}>
            {key.replace("mean_q_", "").toUpperCase()}: {Number(value).toFixed(3)}
          </div>
        ));

        const maxQRendered = Object.entries(maxQPotentials).map(([key, value]) => (
          <div key={`maxqpot-${data.epoch}-${key}`} style={{ fontSize: "0.9em", color: "#D81B60", paddingLeft: "8px" }}>
            {key}: {Number(value).toFixed(3)}
          </div>
        ));

        const formattedOps = Object.entries(opProbs).map(([op, prob]) => {
          const isChosen = op === data.chosen_operation;
          return (
            <div
              key={`op-prob-${data.epoch}-${op}`}
              style={{
                fontSize: "0.9em",
                color: isChosen ? "#2E7D32" : "#333",
                fontWeight: isChosen ? "bold" : "normal",
                paddingLeft: "8px",
              }}
            >
              {isChosen ? "✓" : ""} {op} (π = {Number(prob).toFixed(3)})
            </div>
          );
        });

        const topKByOp = data.top_k_q_by_op || {};
        const qValuesForChosenOp = topKByOp[data.chosen_operation] || [];
        const qValuesRendered = qValuesForChosenOp.map((entry) => {
          const isChosen = entry[0] === data.chosen_feature;
          return (
            <div
              key={`topq-${data.epoch}-${entry[0]}`}
              style={{
                fontSize: "0.9em",
                color: isChosen ? "#2E7D32" : "#444",
                fontWeight: isChosen ? "bold" : "normal",
                paddingLeft: "8px",
              }}
            >
              {isChosen ? "✓" : "-"} Feature {entry[0]} (Q = {Number(entry[1]).toFixed(3)})
            </div>
          );
        });

        setPolicyLogs(prevLogs => [
          ...prevLogs,
          <div key={`decision-${data.epoch}`} style={{ borderTop: "1px solid #eee", paddingTop: "8px", marginBottom: "8px" }}>
            <div style={{ fontWeight: "bold" }}>Epoch {data.epoch} Decision:</div>
            <div><strong>Mean Q-Values:</strong>{meanQRendered}</div>
            <div><strong>Max Q-Potentials:</strong>{maxQRendered}</div>
            <div><strong>Operation Probs:</strong>{formattedOps}</div>
            <div><strong>Top Q-Values for {data.chosen_operation}:</strong>{qValuesRendered}</div>
          </div>,
        ]);
      } else if (data.stage === "feature_selection") {
        const { operation, feature } = data.action_taken;
        const features = Array.from(new Set(data.selected_features || []));
        setCurrentFeatureSet(features);

        setPolicyLogs(prevLogs => [
          ...prevLogs,
          <div style={{ color: "#03A9F4" }}>
            Action: <strong>{operation}</strong> feature <strong>{feature}</strong>. New Set: [{features.join(", ")}]
          </div>,
        ]);
      } else if (data.stage === "accuracy") {
        const currentSet = Array.isArray(data.selected_features)
          ? Array.from(new Set(data.selected_features))
          : [];

        const currentF1 = Number(data.f1_score || 0);
        const currentAccuracy = Number(data.accuracy || 0);
        const currentReward = Number(data.reward_total || 0);

        updateTrainingMetrics(
          data.epoch,
          currentAccuracy,
          currentF1,
          currentSet.length,
          currentReward
        );

        const lastLoggedBest = bestPoliciesRef.current[bestPoliciesRef.current.length - 1];
        const isSameAsLastBest =
          !!lastLoggedBest &&
          Math.abs(lastLoggedBest.f1_score - currentF1) <= EPS &&
          lastLoggedBest.features.length === currentSet.length &&
          new Set(lastLoggedBest.features).size === new Set(currentSet).size &&
          lastLoggedBest.features.every(f => currentSet.includes(f));

        const betterF1 = currentF1 > bestF1Run + EPS;
        const tieF1 = Math.abs(currentF1 - bestF1Run) <= EPS;
        const fewerFeat = currentSet.length < bestSizeRun;
        const isNewBest = betterF1 || (tieF1 && fewerFeat);

        if (isNewBest && !isSameAsLastBest) {
          bestF1Run = currentF1;
          bestSizeRun = currentSet.length;
          bestSetRun = [...currentSet];

          setBestF1SoFar(currentF1);
          setBestFeatureCount(currentSet.length);
          setBestFeatures([...currentSet]);

          const newPolicy = { epoch: data.epoch, f1_score: currentF1, features: [...currentSet] };
          bestPoliciesRef.current.push(newPolicy);
          setBestPolicies([...bestPoliciesRef.current]);
        }

        setPolicyLogs(prevLogs => [
          ...prevLogs,
          <div style={{ color: "#4CAF50" }}>
            Result: Accuracy: <strong>{(data.accuracy * 100).toFixed(2)}%</strong> | F1:{" "}
            <strong>{currentF1.toFixed(3)}</strong>
          </div>,
        ]);
      } else if (data.stage === "warmup") {
        const logMessage = `Warmup step (${data.steps_done}/${data.warmup_steps_required})...`;
        setLogs(prevLogs => [
          ...prevLogs,
          <div key={`warmup-${data.steps_done}`} style={{ color: "#757575", fontStyle: "italic", marginBottom: "8px" }}>
            {logMessage}
          </div>,
        ]);
      } else if (data.stage === "sac_update") {
        const losses = data.losses || {};
        const alpha_op = Number(losses.alpha_op || 0).toFixed(3);
        const alpha_feat = Number(losses.alpha_feat || 0).toFixed(3);
        const op_entropy = Number(losses.op_entropy || 0).toFixed(3);
        const feat_entropy = Number(losses.feat_entropy || 0);
        const target_op_entropy = Number(losses.target_entropy_op || 0).toFixed(3);
        const target_feat_entropy = Number(losses.target_entropy_feat || 0).toFixed(3);
        const alpha_loss_op = Number(losses.alpha_loss_op || 0).toFixed(4);
        const alpha_loss_feat = Number(losses.alpha_loss_feat || 0).toFixed(4);

        const sacUpdateLog = (
          <div
            key={`sac-update-${data.epoch}`}
            style={{ color: "#757575", fontStyle: "italic", marginBottom: "8px", borderTop: "1px solid #eee", paddingTop: "8px" }}
          >
            <div><strong>SAC Update (Epoch {data.epoch}):</strong></div>
            <div style={{ paddingLeft: "10px", fontSize: "0.9em" }}>
              <div>Reward: <strong>{Number(data.reward_total || 0).toFixed(3)}</strong></div>
              <div>α (Alphas): <strong>α_op={alpha_op}</strong>, <strong>α_feat={alpha_feat}</strong></div>
              <div>Actor Loss: {Number(losses.actor_loss || 0).toFixed(4)}</div>
              <div>Critic Losses: {Number(losses.critic1_loss || 0).toFixed(4)}, {Number(losses.critic2_loss || 0).toFixed(4)}</div>
              <div style={{ marginTop: "4px" }}>
                <div>Op Entropy: {op_entropy} (Target: {target_op_entropy})</div>
                <div>Feat Entropy: {feat_entropy.toFixed(3)} (Target: {target_feat_entropy})</div>
              </div>
              <div style={{ marginTop: "4px" }}>
                <div>Alpha Loss (Op): {alpha_loss_op}</div>
                <div>Alpha Loss (Feat): {alpha_loss_feat}</div>
              </div>
            </div>
          </div>
        );
        setPolicyLogs(prevLogs => [...prevLogs, sacUpdateLog]);
      } else if (data.stage === "policy_training_complete") {
        const finalFeaturesRaw =
          Array.isArray(data.final_features) && data.final_features.length
            ? data.final_features
            : (data.best_features || []);
        const finalFeatures = Array.from(new Set(finalFeaturesRaw));
        const finalNames = finalFeatures.map(f => featureMap[f] || `F${f}`);

        const epochsFinished =
          (typeof data.total_epochs === "number" ? data.total_epochs : null) ??
          (lastEpochSeen || hyperparameters?.max_epochs || null);

        setPolicyLogs(prevLogs => [
          ...prevLogs,
          <div
            key="all-complete"
            style={{
              color: "#1b5e20",
              fontWeight: 600,
              fontSize: "1.0em",
              marginTop: "10px",
              padding: "10px",
              backgroundColor: "#e8f5e9",
              borderRadius: "4px",
              borderLeft: "4px solid #2e7d32"
            }}
          >
            <div style={{ fontSize: "1.1em", marginBottom: "8px" }}>
              <FontAwesomeIcon icon={faCheckCircle} style={{ marginRight: "8px" }} />
              The training have been completed after {epochsFinished} epochs.
            </div>
            <div style={{ marginBottom: "6px" }}>
              Selected feature(s) for training:
            </div>
            <div style={{
              fontFamily: "monospace",
              backgroundColor: "#f1f8e9",
              padding: "8px",
              borderRadius: "4px",
              fontSize: "0.9em"
            }}>
              {finalNames.join(", ")}
            </div>
          </div>,
        ]);

        setBestFeatures(finalFeatures);

        trainingCompleted = true;
        setIsProcessing(false);
        setPolicyTrained(true);
        setWorkAreaMessage("Policy training completed. You can now test the policy or retrain.");
        eventSource.close();

        fetchBestPolicies();
      }
    } catch (err) {
      console.error("Error processing SSE message:", err);
    }
  };

  eventSource.onerror = (err) => {
    if (!trainingCompleted) {
      setLogs(prevLogs => [
        ...prevLogs,
        <div key={`conn-err-${Date.now()}`} style={{ color: "red" }}>
          Connection error. Training halted.
        </div>,
      ]);
      console.error("EventSource failed:", err);
      setIsProcessing(false);
    }
    eventSource.close();
  };
};

const handlePolicyTest = async () => {
  setTestingLogs(["Starting final policy evaluation on the unseen test data..."]);
  setIsProcessing(true);
  setTestResults(null);
  setTestingComplete(false);

  if (!selectedDataset || bestFeatures.length === 0) {
    setTestingLogs(prev => [...prev, "Error: A dataset must be selected and a policy must be trained first."]);
    setIsProcessing(false);
    return;
  }

  if (!selectedClassifier) {
    setTestingLogs(prev => [...prev, "Error: No classifier selected."]);
    setIsProcessing(false);
    return;
  }

  try {
    const classifierMap = { DT: "dt", LR: "logreg", KNN: "knn", RF: "rf", SVM: "svm" };
    const backendClassifier = classifierMap[selectedClassifier];
    if (!backendClassifier) {
      setTestingLogs(prev => [...prev, "Error: Invalid classifier selected for backend."]);
      setIsProcessing(false);
      return;
    }

    const classifier_spec = {
      name: backendClassifier,
      params: {}
    };

    const response = await fetch("http://localhost:5000/test_policy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: selectedDataset,
        features: bestFeatures,
        classifier_spec
      })
    });

    if (!response.ok) {
      let msg = "An unknown error occurred during testing.";
      try {
        const err = await response.json();
        msg = err.error || msg;
      } catch {}
      throw new Error(msg);
    }

    const results = await response.json();

    setTestResults({
      policy_performance: results.policy_performance,
      all_features_performance: results.all_features_performance,
      numTestSamples: results.num_test_samples,
      download_available: results.download_available,
      export_endpoint: results.export_endpoint,
      export_payload: results.export_payload
    });

    const policyPerf = results.policy_performance || {};
    const allFeaturesPerf = results.all_features_performance || {};
    const feats = Array.isArray(policyPerf.features_tested)
      ? policyPerf.features_tested.join(", ")
      : String(policyPerf.features_tested ?? "");

    setTestingLogs(prev => [
      ...prev,
      "--- Evaluation Complete ---",
      `Classifier Used: ${policyPerf.classifier_used}`,
      `Features Tested: [${feats}]`,
      `Policy F1 Score: ${Number(policyPerf.f1_score).toFixed(4)}`,
      `All Features F1 Score: ${Number(allFeaturesPerf.f1_score).toFixed(4)}`,
      `Training Time - Policy: ${policyPerf.training_time}s, All Features: ${allFeaturesPerf.training_time}s`,
      `Inference Time - Policy: ${policyPerf.inference_time}s, All Features: ${allFeaturesPerf.inference_time}s`,
      results.download_available ? "✓ Models ready for download" : "⚠ Models not available for download"
    ]);

  } catch (error) {
    setTestingLogs(prev => [...prev, `Error during testing: ${error.message}`]);
  } finally {
    setIsProcessing(false);
    setTestingComplete(true);
  }
};

const handleDownloadModels = async () => {
  if (!testResults || !testResults.download_available) {
    setTestingLogs(prev => [...prev, "Error: Models not available for download"]);
    return;
  }

  try {
    setIsProcessing(true);
    setTestingLogs(prev => [...prev, "Preparing models for download..."]);
    
    const response = await fetch("http://localhost:5000/export_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(testResults.export_payload)
    });

    if (!response.ok) {
      throw new Error("Failed to download models");
    }

    const blob = await response.blob();
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.style.display = 'none';
    a.href = url;
    
    const contentDisposition = response.headers.get('content-disposition');
    let filename = 'iot_ransomware_models.zip';
    if (contentDisposition) {
      const filenameMatch = contentDisposition.match(/filename="(.+)"/);
      if (filenameMatch) {
        filename = filenameMatch[1];
      }
    }
    
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    window.URL.revokeObjectURL(url);
    document.body.removeChild(a);
    
    setTestingLogs(prev => [...prev, `✓ Models downloaded successfully: ${filename}`]);
  } catch (error) {
    setTestingLogs(prev => [...prev, `Error downloading models: ${error.message}`]);
  } finally {
    setIsProcessing(false);
  }
};

  const resetExperiment = () => {
    setCurrentStage("Start");
    setLogs(["System reset. Ready to begin new experiment..."]);
    setPolicyLogs([]);
    setUploadedFile(null);
    setSelectedDataset(null);
    setPreprocessingComplete(false);
    setPolicyTrained(false);
    setTestingComplete(false);
    setBestFeatures([]);
    setBestPolicies([]);
    setTestResults(null);
    setFeatureLookupLogs([]);
    setFeatureMap({});
    setSelectedClassifier("");
    setTestingLogs([]);
    resetTrainingMetrics();
    setShowPerformanceGraph(false);
    setShowFeaturesGraph(false);
  };

  const HyperparameterModal = () => (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0,
      backgroundColor: "rgba(0,0,0,0.5)",
      display: "flex",
      alignItems: "center",
      justifyContent: "center",
      zIndex: 1000
    }}>
      <div style={{
        backgroundColor: "#fff",
        padding: "25px",
        borderRadius: "8px",
        width: "600px",
        maxHeight: "80vh",
        overflowY: "auto"
      }}>
        <h3 style={{ marginTop: 0 }}>Configure Training Hyperparameters</h3>
        
        <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "15px", marginBottom: "20px" }}>
          {Object.entries(hyperparameters).map(([key, value]) => (
            <div key={key}>
              <label style={{ display: "block", marginBottom: "5px", fontWeight: "500" }}>
                {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}:
              </label>
              <input
                type="number"
                step={key.includes('rate') || key.includes('alpha') || key.includes('gamma') || key.includes('tau') || key.includes('ratio') ? "0.0001" : "1"}
                value={value}
                onChange={(e) => setHyperparameters({
                  ...hyperparameters,
                  [key]: key.includes('learning_rate') || key.includes('alpha') || key.includes('gamma') || key.includes('tau') || key.includes('ratio') ? 
                    parseFloat(e.target.value) : parseInt(e.target.value)
                })}
                min={key.includes('ratio') ? 0 : key.includes('rate') ? 0.00001 : 1}
                max={key.includes('ratio') ? 1 : key.includes('rate') ? 0.1 : key === 'gamma' ? 1 : 1000000}
                style={{
                  width: "100%",
                  padding: "8px",
                  borderRadius: "4px",
                  border: `1px solid ${colors.border}`
                }}
              />
            </div>
          ))}
        </div>

        <div style={{ display: "flex", justifyContent: "flex-end", gap: "10px" }}>
          <button 
            onClick={() => setShowHyperparameterModal(false)}
            style={{
              padding: "8px 16px",
              backgroundColor: "#f1f3f4",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
          >
            Cancel
          </button>
          <button 
            onClick={() => {
              setShowHyperparameterModal(false);
              setLogs(prev => [...prev, "Hyperparameters updated"]);
            }}
            style={{
              padding: "8px 16px",
              backgroundColor: colors.primary,
              color: "#fff",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer"
            }}
            >
            Save Configuration
          </button>
        </div>
      </div>
    </div>
  );

  const exportChartsAsImage = () => {
    alert("Chart export functionality would save the graphs as PNG images for your research paper");
  };

  const showPerformanceGraphHandler = () => {
  setShowPerformancePopup(true);
  };

  const showFeaturesGraphHandler = () => {
    setShowFeaturesPopup(true);
  };

  const closeGraphs = () => {
    setShowPerformanceGraph(false);
    setShowFeaturesGraph(false);
  };
const PerformancePopup = () => (
  <div style={{
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.7)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000
  }}>
    <div style={{
      backgroundColor: "#fff",
      padding: "25px",
      borderRadius: "8px",
      width: "80%",
      height: "80%",
      display: "flex",
      flexDirection: "column"
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "20px"
      }}>
        <h3 style={{ margin: 0, color: colors.text }}>Training Performance Metrics</h3>
        <button
          onClick={() => setShowPerformancePopup(false)}
          style={{
            background: "transparent",
            border: "none",
            color: colors.textSecondary,
            cursor: "pointer",
            fontSize: "24px",
            padding: "5px 10px"
          }}
        >
          ×
        </button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        {trainingMetrics.epochs.length > 0 ? (
          <div style={{ height: "100%" }}>
            <Line data={accuracyChartData} options={{
              ...accuracyChartOptions,
              maintainAspectRatio: false
            }} />
          </div>
        ) : (
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            color: colors.textSecondary
          }}>
            No training data available to display
          </div>
        )}
      </div>
      <div style={{ marginTop: "20px", display: "flex", gap: "10px", justifyContent: "flex-end" }}>
        <button
          onClick={exportChartsAsImage}
          style={{
            padding: "8px 16px",
            backgroundColor: colors.primary,
            color: "#fff",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Export as Image
        </button>
        <button
          onClick={() => setShowPerformancePopup(false)}
          style={{
            padding: "8px 16px",
            backgroundColor: colors.border,
            color: colors.text,
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Close
        </button>
      </div>
    </div>
  </div>
);

const FeaturesPopup = () => (
  <div style={{
    position: "fixed",
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: "rgba(0,0,0,0.7)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000
  }}>
    <div style={{
      backgroundColor: "#fff",
      padding: "25px",
      borderRadius: "8px",
      width: "80%",
      height: "80%",
      display: "flex",
      flexDirection: "column"
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "20px"
      }}>
        <h3 style={{ margin: 0, color: colors.text }}>Feature Selection Progress</h3>
        <button
          onClick={() => setShowFeaturesPopup(false)}
          style={{
            background: "transparent",
            border: "none",
            color: colors.textSecondary,
            cursor: "pointer",
            fontSize: "24px",
            padding: "5px 10px"
          }}
        >
          ×
        </button>
      </div>
      <div style={{ flex: 1, minHeight: 0 }}>
        {trainingMetrics.epochs.length > 0 ? (
          <div style={{ height: "100%" }}>
            <Line data={featureChartData} options={{
              ...featureChartOptions,
              maintainAspectRatio: false
            }} />
          </div>
        ) : (
          <div style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            height: "100%",
            color: colors.textSecondary
          }}>
            No training data available to display
          </div>
        )}
      </div>
      <div style={{ marginTop: "20px", display: "flex", gap: "10px", justifyContent: "flex-end" }}>
        <button
          onClick={exportChartsAsImage}
          style={{
            padding: "8px 16px",
            backgroundColor: colors.primary,
            color: "#fff",
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Export as Image
        </button>
        <button
          onClick={() => setShowFeaturesPopup(false)}
          style={{
            padding: "8px 16px",
            backgroundColor: colors.border,
            color: colors.text,
            border: "none",
            borderRadius: "4px",
            cursor: "pointer"
          }}
        >
          Close
        </button>
      </div>
    </div>
  </div>
);

  return (
    <div style={{ 
      display: "flex",
      minHeight: "100vh",
      backgroundColor: colors.background,
      fontFamily: "'Roboto', sans-serif"
    }}>
      {/* Vertical Sidebar Navigation */}
      <div style={{
        width: "220px",
        backgroundColor: colors.sidebar,
        boxShadow: "2px 0 10px rgba(0,0,0,0.1)",
        padding: "20px 0",
        display: "flex",
        flexDirection: "column",
        zIndex: 100
      }}>
        <div style={{
          padding: "0 20px 20px",
          borderBottom: `1px solid ${colors.border}`
        }}>
          <h2 style={{
            margin: 0,
            color: colors.primary,
            fontSize: "18px",
            fontWeight: "500",
            display: "flex",
            alignItems: "center",
            gap: "10px"
          }}>
            <FontAwesomeIcon icon={faFlask} />
            <span>DRL IoT Ransomware Detection Research System</span>
          </h2>
        </div>

        <div style={{
          flex: 1,
          overflowY: "auto",
          padding: "20px 0"
        }}>
          {stages.map((stage) => (
            <div
              key={stage.name}
              onClick={() => setCurrentStage(stage.name)}
              style={{
                padding: "12px 20px",
                margin: "5px 0",
                cursor: "pointer",
                backgroundColor: currentStage === stage.name ? colors.primary : "transparent",
                color: currentStage === stage.name ? "#fff" : colors.text,
                borderLeft: currentStage === stage.name ? `4px solid ${colors.primary}` : "4px solid transparent",
                transition: "all 0.2s ease",
                display: "flex",
                alignItems: "center",
                gap: "12px",
                borderRadius: "0 4px 4px 0"
              }}
            >
              <span style={{ fontSize: "16px" }}>{stage.icon}</span>
              <div>
                <div style={{ fontWeight: "500" }}>{stage.name}</div>
                <div style={{ 
                  fontSize: "12px", 
                  color: currentStage === stage.name ? "rgba(255,255,255,0.8)" : colors.textSecondary 
                }}>
                  {stage.description}
                </div>
              </div>
            </div>
          ))}
        </div>

        <div style={{
          padding: "20px",
          borderTop: `1px solid ${colors.border}`,
          color: colors.textSecondary,
          fontSize: "12px"
        }}>
          <div>System v1.0</div>
          <div>By Anthony Nchabeleng</div>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        overflow: "hidden"
      }}>
        {/* Top Bar */}
        <div style={{
          backgroundColor: colors.sidebar,
          padding: "15px 25px",
          boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center"
        }}>
          <h2 style={{
            margin: 0,
            color: colors.text,
            fontSize: "20px",
            fontWeight: "500"
          }}>
            {currentStage === "Start" ? "Experiment Setup" : currentStage}
          </h2>
          <div style={{ color: colors.textSecondary }}>
            {new Date().toLocaleDateString()}
          </div>
        </div>

        {/* Work Area */}
        <div style={{
          flex: 1,
          padding: "25px",
          overflowY: "auto"
        }}>
          {/* Start Screen */}
          {currentStage === "Start" && (
            <div style={{
              maxWidth: "600px",
              margin: "0 auto",
              textAlign: "center",
              padding: "40px 0"
            }}>
              <h1 style={{
                color: colors.primary,
                fontSize: "28px",
                marginBottom: "20px"
              }}>
                IoT Ransomware Detection
              </h1>
              <p style={{
                color: colors.text,
                fontSize: "16px",
                marginBottom: "30px",
                lineHeight: "1.6"
              }}>
                This system implements reinforcement learning for automated feature selection in IoT ransomware detection.
              </p>
              <button
                onClick={() => setCurrentStage("Data")}
                style={{
                  padding: "12px 24px",
                  backgroundColor: colors.primary,
                  color: "#fff",
                  border: "none",
                  borderRadius: "4px",
                  fontSize: "16px",
                  cursor: "pointer",
                  transition: "background-color 0.2s",
                  boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "10px"
                }}
              >
                <FontAwesomeIcon icon={faFlask} />
                Start Experiment
              </button>
            </div>
          )}

          {/* Data Stage */}
          {currentStage === "Data" && (
            <div style={{
              maxWidth: "600px",
              margin: "0 auto",
              backgroundColor: colors.card,
              borderRadius: "8px",
              padding: "25px",
              boxShadow: "0 2px 10px rgba(0,0,0,0.05)"
            }}>
              <h3 style={{
                marginTop: 0,
                marginBottom: "20px",
                color: colors.text
              }}>
                Dataset Selection
              </h3>
              
              <div style={{ marginBottom: "20px" }}>
                <label style={{
                  display: "block",
                  marginBottom: "8px",
                  color: colors.text,
                  fontWeight: "500"
                }}>
                  Select Existing Dataset:
                </label>
                <select
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  style={{
                    width: "100%",
                    padding: "10px",
                    borderRadius: "4px",
                    border: `1px solid ${colors.border}`,
                    backgroundColor: colors.sidebar,
                    color: colors.text
                  }}
                >
                  <option value="">-- Select Dataset --</option>
                  {datasets.map((dataset) => (
                    <option key={dataset.id} value={dataset.id}>
                      {dataset.filename}
                    </option>
                  ))}
                </select>
              </div>
              
              <div style={{ marginBottom: "25px" }}>
                <label style={{
                  display: "block",
                  marginBottom: "8px",
                  color: colors.text,
                  fontWeight: "500"
                }}>
                  Or Upload New Dataset:
                </label>
                <div style={{
                  border: `2px dashed ${colors.border}`,
                  borderRadius: "4px",
                  padding: "20px",
                  textAlign: "center",
                  cursor: "pointer",
                  backgroundColor: uploadedFile ? "#e8f0fe" : colors.sidebar,
                  transition: "all 0.2s"
                }}>
                  <input
                    type="file"
                    onChange={handleFileChange}
                    accept=".csv,.xlsx,.xls"
                    style={{ display: "none" }}
                    id="dataset-upload"
                  />
                  <label htmlFor="dataset-upload" style={{ cursor: "pointer" }}>
                    {uploadedFile ? (
                      <div>
                        <div style={{ color: colors.primary, marginBottom: "5px" }}>
                          <FontAwesomeIcon icon={faDatabase} size="2x" />
                        </div>
                        <div style={{ fontWeight: "500" }}>{uploadedFile.name}</div>
                        <div style={{ fontSize: "13px", color: colors.textSecondary }}>
                          Click to change file
                        </div>
                      </div>
                    ) : (
                      <div>
                        <div style={{ color: colors.textSecondary, marginBottom: "5px" }}>
                          <FontAwesomeIcon icon={faDatabase} size="2x" />
                        </div>
                        <div>Click to browse or drag file here</div>
                        <div style={{ fontSize: "13px", color: colors.textSecondary }}>
                          CSV or Excel format
                        </div>
                      </div>
                    )}
                  </label>
                </div>
              </div>
              
              <button
                onClick={handleSubmitFile}
                disabled={!selectedDataset && !uploadedFile}
                style={{
                  width: "100%",
                  padding: "12px",
                  backgroundColor: (selectedDataset || uploadedFile) ? colors.primary : colors.border,
                  color: "#fff",
                  border: "none",
                  borderRadius: "4px",
                  fontSize: "16px",
                  cursor: (selectedDataset || uploadedFile) ? "pointer" : "not-allowed",
                  transition: "background-color 0.2s"
                }}
              >
                {isProcessing ? "Processing..." : "Continue to Preprocessing"}
              </button>
            </div>
          )}

          {/* Preprocessing Stage */}
          {currentStage === "Preprocessing" && (
            <div style={{
              maxWidth: "800px",
              margin: "0 auto",
              backgroundColor: colors.card,
              borderRadius: "8px",
              padding: "25px",
              boxShadow: "0 2px 10px rgba(0,0,0,0.05)"
            }}>
              <h3 style={{
                marginTop: 0,
                marginBottom: "20px",
                color: colors.text
              }}>
                Data Preprocessing
              </h3>
              
              <div style={{
                backgroundColor: isProcessing ? "#f8f9fa" : "#e8f5e9",
                border: `1px solid ${
                  isProcessing ? colors.border : "#c8e6c9"
                }`,
                borderRadius: "6px",
                padding: "20px",
                marginBottom: "20px"
              }}>
                <div style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "15px",
                  marginBottom: "15px"
                }}>
                  <div style={{
                    width: "40px",
                    height: "40px",
                    borderRadius: "50%",
                    backgroundColor: isProcessing ? "#f1f3f4" : "#e6f4ea",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    color: isProcessing ? colors.textSecondary : colors.success
                  }}>
                    <FontAwesomeIcon icon={isProcessing ? faTools : faCheckCircle} />
                  </div>
                  <div>
                    <div style={{ fontWeight: "500", color: colors.text }}>
                      {isProcessing ? "Preprocessing in progress..." : "Preprocessing complete"}
                    </div>
                    <div style={{ fontSize: "14px", color: colors.textSecondary }}>
                      {isProcessing ? "Please wait while we prepare your data" : "Your data is ready for the next step"}
                    </div>
                  </div>
                </div>
                
                {!isProcessing && (
                  <button
                    onClick={() => setCurrentStage("Policy Training")}
                    style={{
                      padding: "10px 15px",
                      backgroundColor: colors.primary,
                      color: "#fff",
                      border: "none",
                      borderRadius: "4px",
                      fontSize: "14px",
                      cursor: "pointer",
                      display: "inline-flex",
                      alignItems: "center",
                      gap: "8px"
                    }}
                  >
                    <FontAwesomeIcon icon={faRobot} />
                    Continue to Policy Training
                  </button>
                )}
              </div>
              
              <div style={{
                backgroundColor: colors.sidebar,
                border: `1px solid ${colors.border}`,
                borderRadius: "6px",
                padding: "20px"
              }}>
                <h4 style={{
                  marginTop: 0,
                  marginBottom: "15px",
                  color: colors.text,
                  fontSize: "16px"
                }}>
                  Processing Log
                </h4>
                <div style={{
                  height: "200px",
                  overflowY: "auto",
                  backgroundColor: colors.background,
                  padding: "10px",
                  borderRadius: "4px",
                  fontFamily: "monospace",
                  fontSize: "13px"
                }}>
                  {logs.map((log, index) => (
                    <div key={index} style={{ marginBottom: "5px" }}>{log}</div>
                  ))}
                </div>
              </div>
            </div>
          )}

      
          {/* Policy Training Stage */}
          {currentStage === "Policy Training" && (
            <div style={{ 
              display: "flex", 
              gap: "20px",
              height: "calc(100vh - 180px)"
            }}>
              {/* Left Panel - Feature Lookup */}
              <div style={{ 
                flex: 1,
                display: "flex",
                flexDirection: "column",
                minHeight: 0
              }}>
                <div style={{ 
                  backgroundColor: colors.card,
                  borderRadius: "8px",
                  padding: "20px",
                  boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  overflow: "hidden"
                }}>
                  <h3 style={{
                    marginTop: 0,
                    marginBottom: "15px",
                    color: colors.text,
                    display: "flex",
                    alignItems: "center",
                    gap: "10px"
                  }}>
                    <FontAwesomeIcon icon={faSearch} />
                    <span>Feature Lookup</span>
                  </h3>
                  <div style={{ 
                    flex: 1,
                    overflowY: "auto",
                    backgroundColor: colors.sidebar,
                    padding: "15px",
                    borderRadius: "4px",
                    border: `1px solid ${colors.border}`,
                    fontFamily: "monospace",
                    fontSize: "13px",
                    minHeight: 0
                  }}>
                    {featureMap && currentFeatureSet.length > 0 ? (
                      <div style={{
                        display: "flex",
                        flexDirection: "column",
                        gap: "10px"
                      }}>
                        {currentFeatureSet.map((featureIndex, i) => (
                          <div key={featureIndex} style={{
                            backgroundColor: i % 2 === 0 ? colors.background : "transparent",
                            borderLeft: `3px solid ${colors.primary}`,
                            borderRadius: "4px",
                            padding: "10px"
                          }}>
                            <div style={{ fontWeight: "500" }}>Feature {featureIndex}:</div>
                            <div style={{
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis"
                            }}>
                              {featureMap[featureIndex] || "Unknown Feature"}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        height: "100%",
                        color: colors.textSecondary
                      }}>
                        {selectedDataset ? "Waiting for feature updates..." : "No dataset selected"}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Center Panel - Training Controls + Log */}
              <div style={{
                flex: 3,
                display: "flex",
                flexDirection: "column",
                minHeight: 0,
                gap: "16px"
              }}>
                {/* Training Controls */}
                <div style={{
                  backgroundColor: colors.card,
                  borderRadius: "8px",
                  padding: "20px",
                  boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                  flex: "0 0 auto"
                }}>
                  <h3 style={{ marginTop: 0, marginBottom: "16px", color: colors.text }}>
                    Policy Training
                  </h3>

                  {/* ML Classifier Selection */}
                  <div style={{ marginBottom: "16px" }}>
                    <label style={{
                      display: "block",
                      marginBottom: "8px",
                      color: colors.text,
                      fontWeight: "500"
                    }}>
                      Select ML Classifier:
                    </label>

                    {selectedClassifier ? (
                      <div style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        padding: "10px 12px",
                        borderRadius: "4px",
                        border: `1px solid ${colors.border}`,
                        backgroundColor: "#f8f9fa"
                      }}>
                        <div style={{ fontSize: "14px", color: colors.text }}>
                          <strong>Selected:</strong>{" "}
                          {classifiers.find(c => c.value === selectedClassifier)?.label}
                        </div>
                        {!isProcessing && !policyTrained && (
                          <button
                            onClick={() => setSelectedClassifier("")}
                            style={{
                              background: "transparent",
                              border: "none",
                              color: colors.primary,
                              cursor: "pointer",
                              fontSize: "13px"
                            }}
                            title="Change classifier"
                          >
                            Change
                          </button>
                        )}
                      </div>
                    ) : (
                      <select
                        value={selectedClassifier}
                        onChange={(e) => setSelectedClassifier(e.target.value)}
                        disabled={isProcessing || policyTrained}
                        style={{
                          width: "100%",
                          padding: "10px",
                          borderRadius: "4px",
                          border: `1px solid ${colors.border}`,
                          backgroundColor: colors.sidebar,
                          color: colors.text,
                          cursor: (isProcessing || policyTrained) ? "not-allowed" : "pointer"
                        }}
                      >
                        <option value="">-- Select Classifier --</option>
                        {classifiers.map((classifier) => (
                          <option key={classifier.value} value={classifier.value}>
                            {classifier.label}
                          </option>
                        ))}
                      </select>
                    )}
                  </div>

                  {/* Status + Controls */}
                  <div style={{
                    backgroundColor: isProcessing ? "#f8f9fa"
                                  : policyTrained ? "#e8f5e9"
                                  : "#e8f4f8",
                    border: `1px solid ${
                      isProcessing ? colors.border
                      : policyTrained ? "#c8e6c9"
                      : "#bbdefb"
                    }`,
                    borderRadius: "6px",
                    padding: "16px"
                  }}>
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      gap: "12px",
                      marginBottom: "12px"
                    }}>
                      <div style={{
                        width: "40px",
                        height: "40px",
                        borderRadius: "50%",
                        backgroundColor: isProcessing ? "#f1f3f4"
                                      : policyTrained ? "#e6f4ea"
                                      : "#e1f5fe",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: isProcessing ? colors.textSecondary
                            : policyTrained ? colors.success
                            : colors.primary
                      }}>
                        <FontAwesomeIcon icon={
                          isProcessing ? faChartLine
                          : policyTrained ? faCheckCircle
                          : faInfoCircle
                        } />
                      </div>
                      <div>
                        <div style={{ fontWeight: "500", color: colors.text }}>
                          {isProcessing ? "Training in progress..."
                          : policyTrained ? "Training complete"
                          : "Ready for training"}
                        </div>
                        <div style={{ fontSize: "14px", color: colors.textSecondary }}>
                          {isProcessing ? "Training the SAC policy model"
                          : policyTrained ? "Model is ready for testing"
                          : "Configure and train the reinforcement learning policy"}
                        </div>
                      </div>
                    </div>

                    {/* Action buttons */}
                    <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                      {!policyTrained && (
                        <>
                          <button
                            onClick={() => setShowHyperparameterModal(true)}
                            disabled={isProcessing}
                            style={{
                              padding: "10px 15px",
                              backgroundColor: isProcessing ? colors.border : colors.primary,
                              color: "#fff",
                              border: "none",
                              borderRadius: "4px",
                              fontSize: "14px",
                              cursor: isProcessing ? "not-allowed" : "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "8px"
                            }}
                          >
                            <FontAwesomeIcon icon={faCog} />
                            Configure Hyperparameters
                          </button>

                          <button
                            onClick={handlePolicyTraining}
                            disabled={isProcessing || !preprocessingComplete || !selectedClassifier}
                            style={{
                              padding: "10px 15px",
                              backgroundColor:
                                (isProcessing || !preprocessingComplete || !selectedClassifier)
                                  ? colors.border
                                  : colors.primary,
                              color: "#fff",
                              border: "none",
                              borderRadius: "4px",
                              fontSize: "14px",
                              cursor:
                                (isProcessing || !preprocessingComplete || !selectedClassifier)
                                  ? "not-allowed"
                                  : "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "8px"
                            }}
                          >
                            <FontAwesomeIcon icon={isProcessing ? faSpinner : faChartLine} spin={isProcessing} />
                            {isProcessing ? "Training..." : "Train Policy Model"}
                          </button>
                        </>
                      )}

                      {policyTrained && (
                        <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", width: "100%" }}>
                          <button
                            onClick={() => {
                              setCurrentStage("Policy Testing");
                              setTestingComplete(false);
                              setTestResults(null);
                              setTestingLogs([
                                "Ready for policy testing. Click the button below to begin testing."
                              ]);
                            }}
                            style={{
                              padding: "10px 15px",
                              backgroundColor: colors.success,
                              color: "#fff",
                              border: "none",
                              borderRadius: "4px",
                              fontSize: "14px",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "8px"
                            }}
                          >
                            <FontAwesomeIcon icon={faVial} />
                            Proceed to Policy Testing
                          </button>
                          
                          {/* Updated buttons for popup windows */}
                          <button
                            onClick={showPerformanceGraphHandler}
                            style={{
                              padding: "10px 15px",
                              backgroundColor: colors.primary,
                              color: "#fff",
                              border: "none",
                              borderRadius: "4px",
                              fontSize: "14px",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "8px"
                            }}
                          >
                            <FontAwesomeIcon icon={faChartBar} />
                            Performance
                          </button>
                          
                          <button
                            onClick={showFeaturesGraphHandler}
                            style={{
                              padding: "10px 15px",
                              backgroundColor: colors.primary,
                              color: "#fff",
                              border: "none",
                              borderRadius: "4px",
                              fontSize: "14px",
                              cursor: "pointer",
                              display: "inline-flex",
                              alignItems: "center",
                              gap: "8px"
                            }}
                          >
                            <FontAwesomeIcon icon={faChartArea} />
                            Features
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Training Log — Now takes majority of vertical space */}
                <div style={{
                  flex: 1,
                  backgroundColor: colors.sidebar,
                  border: `1px solid ${colors.border}`,
                  borderRadius: "6px",
                  padding: "20px",
                  display: "flex",
                  flexDirection: "column",
                  minHeight: 0,
                  overflow: "hidden"
                }}>
                  <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "12px",
                    flex: "0 0 auto"
                  }}>
                    <h4 style={{ marginTop: 0, color: colors.text, fontSize: "16px" }}>
                      Training Log
                    </h4>
                    {policyTrained && (
                      <div style={{
                        backgroundColor: "#e6f4ea",
                        color: colors.success,
                        padding: "5px 10px",
                        borderRadius: "4px",
                        fontSize: "13px",
                        fontWeight: "500"
                      }}>
                        Training Complete
                      </div>
                    )}
                  </div>

                  <div style={{
                    flex: 1,
                    overflowY: "auto",
                    backgroundColor: colors.background,
                    padding: "16px",
                    borderRadius: "4px",
                    fontFamily: "monospace",
                    fontSize: "13px",
                    minHeight: 0,
                    maxHeight: "100%",
                    border: `1px solid ${colors.border}`
                  }}>
                    {policyLogs.length === 0 ? (
                      <div style={{ 
                        display: "flex", 
                        alignItems: "center", 
                        justifyContent: "center", 
                        height: "100%",
                        color: colors.textSecondary,
                        fontStyle: "italic"
                      }}>
                        {isProcessing ? "Training in progress... logs will appear here" : "No training logs yet"}
                      </div>
                    ) : (
                      policyLogs.map((log, index) => (
                        <div key={index} style={{ 
                          marginBottom: "12px",
                          padding: "8px",
                          backgroundColor: index % 2 === 0 ? "rgba(0,0,0,0.02)" : "transparent",
                          borderRadius: "4px"
                        }}>
                          {log}
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {/* Right Panel - Best Policies */}
              <div style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                minHeight: 0
              }}>
                <div style={{
                  backgroundColor: colors.card,
                  borderRadius: "8px",
                  padding: "20px",
                  boxShadow: "0 2px 10px rgba(0,0,0,0.05)",
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  overflow: "hidden"
                }}>
                  <h3 style={{
                    marginTop: 0,
                    marginBottom: "15px",
                    color: colors.text,
                    display: "flex",
                    alignItems: "center",
                    gap: "10px"
                  }}>
                    <FontAwesomeIcon icon={faMedal} />
                    <span>Best Policies</span>
                  </h3>

                  <div style={{
                    flex: 1,
                    overflowY: "auto",
                    backgroundColor: colors.sidebar,
                    padding: "15px",
                    borderRadius: "4px",
                    border: `1px solid ${colors.border}`,
                    minHeight: 0
                  }}>
                    {bestPolicies.length > 0 ? (
                      bestPolicies.map((policy, index) => {
                        const mapped = policy.features.map(f => featureMap[f] || `F${f}`);
                        return (
                          <div key={index} style={{
                            marginBottom: "12px",
                            padding: "14px",
                            backgroundColor: "#f9f9f9",
                            borderRadius: "6px",
                            borderLeft: `4px solid ${colors.primary}`,
                            boxShadow: "0 1px 3px rgba(0,0,0,0.08)"
                          }}>
                            <div style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "center",
                              marginBottom: "6px"
                            }}>
                              <div style={{
                                fontWeight: "600",
                                fontSize: "15px",
                                color: colors.primary
                              }}>
                                Epoch {policy.epoch}
                              </div>
                              <div style={{
                                backgroundColor: "#e6f4ea",
                                color: colors.success,
                                padding: "3px 8px",
                                borderRadius: "4px",
                                fontSize: "12px",
                                fontWeight: "500"
                              }}>
                                F1: {policy.f1_score.toFixed(3)}
                              </div>
                            </div>
                            <div style={{
                              color: colors.textSecondary,
                              fontSize: "12px",
                              marginBottom: "4px"
                            }}>
                              Features:
                            </div>
                            <div style={{
                              backgroundColor: colors.background,
                              padding: "8px",
                              borderRadius: "4px",
                              fontFamily: "monospace",
                              fontSize: "12px",
                              color: colors.text
                            }}>
                              [{mapped.join(', ')}]
                            </div>
                          </div>
                        );
                      })
                    ) : (
                      <div style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        height: "100%",
                        color: colors.textSecondary,
                        textAlign: "center"
                      }}>
                        {isProcessing ? (
                          <div>
                            <FontAwesomeIcon icon={faSpinner} spin style={{ marginBottom: "10px", fontSize: "20px" }} />
                            <div>Training in progress...</div>
                          </div>
                        ) : (
                          "No policies discovered yet"
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {showHyperparameterModal && <HyperparameterModal />}
              {showPerformancePopup && <PerformancePopup />}
              {showFeaturesPopup && <FeaturesPopup />}
            </div>
          )}
          {/* Policy Testing Stage */}
{currentStage === "Policy Testing" && (
  <div style={{
    maxWidth: "800px",
    margin: "0 auto",
    backgroundColor: colors.card,
    borderRadius: "8px",
    padding: "20px",
    boxShadow: "0 2px 10px rgba(0,0,0,0.05)"
  }}>
    <h3 style={{
      marginTop: 0,
      marginBottom: "15px",
      color: colors.text
    }}>
      Policy Testing
    </h3>
    
    <div style={{
      backgroundColor: isProcessing ? "#f8f9fa" : 
                      testingComplete ? "#e8f5e9" : "#fff8e1",
      border: `1px solid ${
        isProcessing ? colors.border : 
        testingComplete ? "#c8e6c9" : "#ffe082"
      }`,
      borderRadius: "6px",
      padding: "15px",
      marginBottom: "15px"
    }}>
      <div style={{
        display: "flex",
        alignItems: "center",
        gap: "12px",
        marginBottom: testingComplete ? "0" : "12px"
      }}>
        <div style={{
          width: "32px",
          height: "32px",
          borderRadius: "50%",
          backgroundColor: isProcessing ? "#f1f3f4" : 
                            testingComplete ? "#e6f4ea" : "#fff3bf",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          color: isProcessing ? colors.textSecondary : 
                testingComplete ? colors.success : colors.warning
        }}>
          <FontAwesomeIcon icon={
            isProcessing ? faVial : 
            testingComplete ? faCheckCircle : faInfoCircle
          } />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: "500", color: colors.text }}>
            {isProcessing ? "Policy testing in progress..." : 
            testingComplete ? "Policy testing complete" : "Ready for policy testing"}
          </div>
          <div style={{ fontSize: "13px", color: colors.textSecondary }}>
            {isProcessing ? "Evaluating the trained policy model" : 
            testingComplete ? "View test results below" : 
            "Click the button below to begin testing"}
          </div>
        </div>
        
        {!testingComplete && (
          <button
            onClick={handlePolicyTest}
            disabled={isProcessing || !policyTrained}
            style={{
              padding: "8px 12px",
              backgroundColor: isProcessing || !policyTrained ? colors.border : colors.primary,
              color: "#fff",
              border: "none",
              borderRadius: "4px",
              fontSize: "13px",
              cursor: isProcessing || !policyTrained ? "not-allowed" : "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: "6px",
              whiteSpace: "nowrap"
            }}
          >
            <FontAwesomeIcon icon={isProcessing ? faSpinner : faVial} spin={isProcessing} />
            {isProcessing ? "Testing..." : "Test Policy"}
          </button>
        )}
      </div>
      
      {testingComplete && testResults && (
        <div style={{
          marginTop: "15px",
          padding: "12px",
          backgroundColor: "#f8f9fa",
          borderRadius: "4px",
          borderLeft: `3px solid ${colors.success}`
        }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "12px" }}>
            <h4 style={{ margin: 0, color: colors.text, fontSize: "15px" }}>
              Test Results Summary
            </h4>
            <div style={{ display: "flex", gap: "10px" }}>
              {/* NEW: Download Models Button */}
              {testResults.download_available && (
                <button
                  onClick={handleDownloadModels}
                  disabled={isProcessing}
                  style={{
                    padding: "5px 10px",
                    backgroundColor: colors.success,
                    color: "#fff",
                    border: "none",
                    borderRadius: "4px",
                    fontSize: "12px",
                    cursor: isProcessing ? "not-allowed" : "pointer",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "5px"
                  }}
                >
                  <FontAwesomeIcon icon={isProcessing ? faSpinner : faDownload} spin={isProcessing} />
                  {isProcessing ? "Preparing..." : "Download Models"}
                </button>
              )}
              <button
                onClick={resetExperiment}
                style={{
                  padding: "5px 10px",
                  backgroundColor: colors.primary,
                  color: "#fff",
                  border: "none",
                  borderRadius: "4px",
                  fontSize: "12px",
                  cursor: "pointer",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "5px"
                }}
              >
                <FontAwesomeIcon icon={faArrowLeft} />
                New Experiment
              </button>
            </div>
          </div>
          
          {/* Compact Performance Metrics */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "10px", marginBottom: "12px" }}>
            <div style={{ textAlign: "center", padding: "10px", backgroundColor: "#e8f5e9", borderRadius: "4px" }}>
              <div style={{ fontSize: "12px", color: colors.textSecondary }}>Policy Accuracy</div>
              <div style={{ fontSize: "18px", fontWeight: "600", color: colors.success }}>
                {Math.round(testResults.policy_performance.accuracy * 100)}%
              </div>
            </div>
            <div style={{ textAlign: "center", padding: "10px", backgroundColor: "#e3f2fd", borderRadius: "4px" }}>
              <div style={{ fontSize: "12px", color: colors.textSecondary }}>All Features Accuracy</div>
              <div style={{ fontSize: "18px", fontWeight: "600", color: colors.primary }}>
                {Math.round(testResults.all_features_performance.accuracy * 100)}%
              </div>
            </div>
          </div>
          
          {/* Compact Timing Comparison */}
          <div style={{ marginBottom: "10px" }}>
            <div style={{ fontSize: "13px", fontWeight: "500", color: colors.text, marginBottom: "8px" }}>
              Computational Efficiency
            </div>
            
            {/* Training Time - Single Row */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
              <span style={{ fontSize: "12px", color: colors.textSecondary }}>Training Time:</span>
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <span style={{ fontSize: "12px", color: colors.success, fontWeight: "500" }}>
                  {testResults.policy_performance.training_time}s
                </span>
                <span style={{ fontSize: "11px", color: colors.textSecondary }}>vs</span>
                <span style={{ fontSize: "12px", color: colors.primary }}>
                  {testResults.all_features_performance.training_time}s
                </span>
                <span style={{ fontSize: "11px", color: colors.success, fontWeight: "500" }}>
                  {((1 - testResults.policy_performance.training_time / testResults.all_features_performance.training_time) * 100).toFixed(1)}% faster
                </span>
              </div>
            </div>
            
            {/* Inference Time - Single Row */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <span style={{ fontSize: "12px", color: colors.textSecondary }}>Inference Time:</span>
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <span style={{ fontSize: "12px", color: colors.success, fontWeight: "500" }}>
                  {testResults.policy_performance.inference_time}s
                </span>
                <span style={{ fontSize: "11px", color: colors.textSecondary }}>vs</span>
                <span style={{ fontSize: "12px", color: colors.primary }}>
                  {testResults.all_features_performance.inference_time}s
                </span>
                <span style={{ fontSize: "11px", color: colors.success, fontWeight: "500" }}>
                  {((1 - testResults.policy_performance.inference_time / testResults.all_features_performance.inference_time) * 100).toFixed(1)}% faster
                </span>
              </div>
            </div>
          </div>

          {/* Compact F1 Score Comparison */}
          <div style={{ 
            display: "flex", 
            justifyContent: "space-between",
            alignItems: "center",
            paddingTop: "10px", 
            borderTop: `1px solid ${colors.border}` 
          }}>
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ fontSize: "12px", color: colors.textSecondary }}>F1 Score:</span>
              <span style={{ fontSize: "12px", color: colors.success, fontWeight: "500" }}>
                {testResults.policy_performance.f1_score.toFixed(4)}
              </span>
              <span style={{ fontSize: "11px", color: colors.textSecondary }}>vs</span>
              <span style={{ fontSize: "12px", color: colors.primary }}>
                {testResults.all_features_performance.f1_score.toFixed(4)}
              </span>
            </div>
            <div style={{ fontSize: "11px", color: colors.textSecondary }}>
              Features: {Array.isArray(testResults.policy_performance.features_tested) 
                ? testResults.policy_performance.features_tested.length 
                : 'N/A'} selected
            </div>
          </div>

          {/* NEW: Download Information */}
          {testResults.download_available && (
            <div style={{
              marginTop: "10px",
              padding: "8px",
              backgroundColor: "#e8f5e9",
              borderRadius: "4px",
              borderLeft: `3px solid ${colors.success}`
            }}>
              <div style={{ fontSize: "12px", color: colors.text, fontWeight: "500" }}>
                <FontAwesomeIcon icon={faDownload} style={{ marginRight: "5px" }} />
                Models Ready for Download
              </div>
              <div style={{ fontSize: "11px", color: colors.textSecondary, marginTop: "2px" }}>
                Click "Download Models" to get both trained models (selected features + all features) as a ZIP file
              </div>
            </div>
          )}
        </div>
      )}
    </div>
    
    {/* Testing Log - Made smaller */}
    <div style={{
      backgroundColor: colors.sidebar,
      border: `1px solid ${colors.border}`,
      borderRadius: "6px",
      padding: "15px"
    }}>
      <div style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        marginBottom: "12px"
      }}>
        <h4 style={{
          marginTop: 0,
          color: colors.text,
          fontSize: "15px"
        }}>
          Testing Log
        </h4>
        {testingComplete && (
          <div style={{
            backgroundColor: "#e6f4ea",
            color: colors.success,
            padding: "4px 8px",
            borderRadius: "4px",
            fontSize: "12px",
            fontWeight: "500"
          }}>
            Complete
          </div>
        )}
      </div>
      <div style={{
        height: "200px", // Reduced from 300px
        overflowY: "auto",
        backgroundColor: colors.background,
        padding: "8px",
        borderRadius: "4px",
        fontFamily: "monospace",
        fontSize: "12px"
      }}>
        {testingLogs.length === 0 ? (
            <div style={{ color: colors.textSecondary }}>
              Waiting to start policy testing…
            </div>
          ) : (
            testingLogs.map((log, index) => (
              <div key={index} style={{ marginBottom: "4px" }}>{log}</div>
            ))
          )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default App;