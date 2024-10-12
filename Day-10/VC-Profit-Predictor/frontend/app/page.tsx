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
import {
  DollarSign,
  TrendingUp,
  Lightbulb,
  ChartNoAxesCombined,
} from "lucide-react";

const sampleData = [
  {
    id: 0,
    RnD_Spend: 165349.2,
    Administration: 136897.8,
    Marketing_Spend: 471784.1,
    State: "new-york",
  },
  {
    id: 1,
    RnD_Spend: 162597.7,
    Administration: 151377.59,
    Marketing_Spend: 443898.53,
    State: "california",
  },
  {
    id: 2,
    RnD_Spend: 153441.51,
    Administration: 101145.55,
    Marketing_Spend: 407934.54,
    State: "florida",
  },
  {
    id: 3,
    RnD_Spend: 144372.41,
    Administration: 118671.85,
    Marketing_Spend: 383199.62,
    State: "new-york",
  },
  {
    id: 4,
    RnD_Spend: 142107.34,
    Administration: 91391.77,
    Marketing_Spend: 366168.42,
    State: "florida",
  },
];

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

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);

    const formData = new FormData(event.currentTarget);
    const data = Object.fromEntries(formData.entries());

    try {
      const response = await fetch(
        "https://api-vc-profit-predictor.vercel.app/api/predict",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(data),
        },
      );

      if (!response.ok) {
        throw new Error("Prediction failed");
      }

      const result = await response.json();
      setPrediction(result.prediction);
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while making the prediction.");
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

  return (
    <div className="flex flex-col min-h-screen">
      <header className="flex top-0 sticky items-center h-16 px-4 border-b shrink-0 md:px-6 bg-white bg-opacity-50 backdrop-blur-md">
        <nav className="flex-col hidden gap-6 text-lg font-medium md:flex md:flex-row md:items-center md:gap-5 md:text-sm lg:gap-6">
          <a
            className="flex items-center gap-2 text-lg font-semibold md:text-base"
            href="/"
          >
            <ChartNoAxesCombined className="w-6 h-6" />
            <span>VC Profit Predictor</span>
          </a>
        </nav>
      </header>

      <main className="flex-1 p-4 md:p-6">
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
            </ul>
            <p className="font-medium">
              This application is created only for learning purposes**
            </p>
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
                      className={`text-xs ${
                        activeSample === null
                          ? "bg-primary text-primary-foreground"
                          : ""
                      }`}
                      onClick={() => {
                        setFormData({
                          RnD_Spend: "",
                          Administration: "",
                          Marketing_Spend: "",
                          State: "",
                        });
                        setActiveSample(null);
                      }}
                    >
                      Custom
                    </Button>
                    {sampleData.map((sample) => (
                      <Button
                        key={sample.id}
                        type="button"
                        variant="outline"
                        className={`text-xs ${
                          activeSample === sample.id
                            ? "bg-primary text-primary-foreground"
                            : ""
                        }`}
                        onClick={() => handleSampleSelect(sample.id)}
                      >
                        Sample {sample.id + 1}
                      </Button>
                    ))}
                  </div>
                </div>
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

      <footer className="py-4 text-center bg-white border-2">
        <p className="text-sm text-black font-medium">
          Created by Trishan Wagle
        </p>
      </footer>
    </div>
  );
}
