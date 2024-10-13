"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  DollarSign,
  TrendingUp,
  Lightbulb,
  FileUp,
  AlertCircle,
  CheckCircle2,
  ChartNoAxesCombined,
} from "lucide-react";
import { TrainModel } from "./TrainModel";
import Link from "next/link";
import { sampleData } from "@/data";

interface AlertMessage {
  type: "error" | "success";
  title: string;
  message: string;
}

export default function ProfitPredictorVC() {
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [formData, setFormData] = useState({
    RnD_Spend: "",
    Administration: "",
    Marketing_Spend: "",
    State: "",
  });
  const [activeSample, setActiveSample] = useState<number | null>(null);
  const [selectedModel, setSelectedModel] = useState("default");
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [isTrainModelOpen, setIsTrainModelOpen] = useState(false);
  const [alertMessage, setAlertMessage] = useState<AlertMessage | null>(null);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setAlertMessage(null);

    const formData = new FormData(event.currentTarget);

    if (selectedModel !== "default" && modelFile) {
      formData.append("model-file", modelFile);
    }

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/predict`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const result = await response.json();
      setPrediction(result.prediction);
      setAlertMessage({
        type: "success",
        title: "Prediction Successful",
        message: "Your profit prediction has been calculated successfully.",
      });
    } catch (error) {
      console.error("Error:", error);
      setAlertMessage({
        type: "error",
        title: "Prediction Failed",
        message:
          "An error occurred while making the prediction. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleSampleSelect = (sampleId: number) => {
    const sample = sampleData.find((s) => s.id === sampleId);
    if (sample) {
      setFormData({
        RnD_Spend: sample.RnD_Spend.toString(),
        Administration: sample.Administration.toString(),
        Marketing_Spend: sample.Marketing_Spend.toString(),
        State: sample.State,
      });
      setActiveSample(sampleId);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFormData((prev) => ({ ...prev, [e.target.name]: e.target.value }));
    setActiveSample(null);
  };

  const handleModelFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setModelFile(e.target.files[0]);
      setSelectedModel("custom");
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      <header className="flex top-0 sticky items-center h-16 px-4 border-b shrink-0 md:px-6 bg-white bg-opacity-50 backdrop-blur-md z-10">
        <nav className="gap-6 text-lg font-medium flex flex-row items-center md:gap-5 md:text-sm lg:gap-6">
          <Link
            className="flex items-center gap-2 text-lg font-semibold md:text-base"
            href="/"
          >
            <ChartNoAxesCombined className="w-6 h-6" />
            <span>VC Profit Predictor</span>
          </Link>

          <Button
            variant="ghost"
            className="text-gray-500 hover:text-gray-900 transition-colors"
            onClick={() => setIsTrainModelOpen(true)}
          >
            <FileUp className="w-4 h-4 mr-2" />
            Train Model
          </Button>
        </nav>
      </header>

      <main className="flex-1 p-4 md:p-6">
        {alertMessage && (
          <Alert
            variant={alertMessage.type === "error" ? "destructive" : "default"}
            className="mb-6"
          >
            {alertMessage.type === "error" ? (
              <AlertCircle className="h-4 w-4" />
            ) : (
              <CheckCircle2 className="h-4 w-4" />
            )}
            <AlertTitle>{alertMessage.title}</AlertTitle>
            <AlertDescription>{alertMessage.message}</AlertDescription>
          </Alert>
        )}

        <Card className="mb-6">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Lightbulb className="w-6 h-6" />
              About This Application
            </CardTitle>
          </CardHeader>

          <CardContent>
            <p className="mb-4">
              Welcome to the VC Profit Predictor! This application is designed
              to help venture capitalists gain insights into potential startup
              profitability based on various spending categories.
            </p>
            <p className="mb-4">
              My machine learning model has been trained on a dataset of
              successful startups. By inputting different spending values, you
              can predict potential profits and understand the impact of various
              financial allocations.
            </p>
            <p>Use this tool to:</p>
            <ul className="list-disc list-inside mb-4">
              <li>
                Analyze how different spending patterns affect profitability
              </li>

              <li>
                Compare the impact of R&D, Administration, and Marketing spend
                on potential profits
              </li>

              <li>
                Understand regional variations in profitability (California, New
                York, Florida)
              </li>

              <li>Develop data-driven investment strategies</li>

              <li>Train custom models using CSV or JSON data files</li>

              <li>Use trained models (.pkl files) for profit predictions</li>
            </ul>
            <p className="mb-4">
              You can train your own model using CSV or JSON data files, which
              will generate a .pkl model file. This custom model can then be
              used for making profit predictions, allowing you to tailor the
              predictions to your specific dataset or industry focus.
            </p>
            <p className="font-medium">
              Note: This application is created for learning purposes only and
              should not be used as the sole basis for investment decisions.
            </p>{" "}
          </CardContent>
        </Card>

        <div className="grid gap-6 md:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Profit Prediction Form</CardTitle>

              <CardDescription>
                Enter startup spending information to predict profit.
              </CardDescription>
            </CardHeader>

            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div className="space-y-2">
                  <Label>Sample Data</Label>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      variant="outline"
                      className={`text-xs focus:bg-primary focus:text-primary-foreground md:focus:text-white md:focus:bg-none ${
                        activeSample === null
                          ? "bg-primary text-primary-foreground"
                          : ""
                      }`}
                      onClick={() => {
                        setActiveSample(null);
                        setFormData({
                          RnD_Spend: "",
                          Administration: "",
                          Marketing_Spend: "",
                          State: "",
                        });
                      }}
                    >
                      Custom
                    </Button>

                    {sampleData.map((sample) => (
                      <Button
                        key={sample.id}
                        type="button"
                        variant="outline"
                        className={`text-xs focus:bg-primary focus:text-primary-foreground md:focus:text-white md:focus:bg-none ${
                          activeSample === sample.id
                            ? "bg-primary text-primary-foreground"
                            : ""
                        }`}
                        onClick={() => {
                          setActiveSample(sample.id);
                          handleSampleSelect(sample.id);
                        }}
                      >
                        Sample {sample.id + 1}
                      </Button>
                    ))}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="select-model">Select Model</Label>

                  <Select
                    value={selectedModel}
                    onValueChange={setSelectedModel}
                  >
                    <SelectTrigger id="select-model">
                      <SelectValue placeholder="Select a model" />
                    </SelectTrigger>

                    <SelectContent>
                      <SelectItem value="default">Default Model</SelectItem>
                      <SelectItem value="custom">Custom Model</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {selectedModel === "custom" && (
                  <div className="space-y-2">
                    <Label htmlFor="model-file">Upload Model File (.pkl)</Label>

                    <Input
                      id="model-file"
                      type="file"
                      accept=".pkl"
                      onChange={handleModelFileChange}
                      required
                    />
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="rd-spend">R&D Spend</Label>

                  <Input
                    id="rd-spend"
                    name="RnD_Spend"
                    placeholder="Enter R&D Spend"
                    type="number"
                    required
                    value={formData.RnD_Spend}
                    onChange={handleInputChange}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="admin-spend">Administration Spend</Label>

                  <Input
                    id="admin-spend"
                    name="Administration"
                    placeholder="Enter Administration Spend"
                    type="number"
                    required
                    value={formData.Administration}
                    onChange={handleInputChange}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="marketing-spend">Marketing Spend</Label>

                  <Input
                    id="marketing-spend"
                    name="Marketing_Spend"
                    placeholder="Enter Marketing Spend"
                    type="number"
                    required
                    value={formData.Marketing_Spend}
                    onChange={handleInputChange}
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="state">State</Label>

                  <Select
                    name="State"
                    value={formData.State}
                    onValueChange={(value) =>
                      setFormData((prev) => ({ ...prev, State: value }))
                    }
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select a state" />
                    </SelectTrigger>

                    <SelectContent>
                      <SelectItem value="california">California</SelectItem>
                      <SelectItem value="new-york">New York</SelectItem>
                      <SelectItem value="florida">Florida</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <Button type="submit" className="w-full" disabled={loading}>
                  {loading ? "Predicting..." : "Predict Profit"}
                </Button>
              </form>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUp className="w-6 h-6" />
                Prediction Result
              </CardTitle>

              <CardDescription>
                The predicted profit based on your inputs.
              </CardDescription>
            </CardHeader>

            <CardContent>
              {prediction !== null ? (
                <div className="space-y-4">
                  <div className="flex items-center justify-center p-6 bg-green-100 rounded-lg">
                    <DollarSign className="w-8 h-8 mr-2 text-green-600" />

                    <span className="text-2xl font-bold text-green-600">
                      {prediction.toLocaleString("en-US", {
                        style: "currency",
                        currency: "USD",
                      })}
                    </span>
                  </div>

                  <p className="text-sm text-gray-600">
                    This prediction suggests the potential profit for a startup
                    with the given spending allocation and location. Use this
                    information to compare different scenarios and identify
                    promising investment opportunities.
                  </p>
                </div>
              ) : (
                <div className="flex items-center justify-center p-6 bg-gray-100 rounded-lg">
                  <span className="text-lg text-gray-500">
                    No prediction yet. Please fill out the form and click
                    &quot;Predict Profit&quot;.
                  </span>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </main>

      <footer className="py-4 text-center bg-white border-t">
        <p className="text-sm text-gray-600">Created by Trishan Wagle</p>
      </footer>

      <TrainModel
        isOpen={isTrainModelOpen}
        onClose={() => setIsTrainModelOpen(false)}
      />
    </div>
  );
}
