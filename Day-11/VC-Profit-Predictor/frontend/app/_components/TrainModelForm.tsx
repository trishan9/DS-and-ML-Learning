import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Download } from "lucide-react";
import { useState } from "react";
import { AlertMessage } from "./TrainModel";

interface TrainModelFormProps {
  file: File | null;
  modelName: string;
  setModelName: React.Dispatch<React.SetStateAction<string>>;
  setAlertMessage: React.Dispatch<React.SetStateAction<AlertMessage | null>>;
  setFile: React.Dispatch<React.SetStateAction<File | null>>;
}

export default function TrainModelForm({
  file,
  modelName,
  setModelName,
  setAlertMessage,
  setFile,
}: TrainModelFormProps) {
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file || !modelName) {
      setAlertMessage({
        type: "error",
        title: "Validation Error",
        message: "Please select a file and enter a model name.",
      });
      return;
    }

    setLoading(true);
    setAlertMessage(null);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("modelName", modelName);

    try {
      const response = await fetch(
        `${process.env.NEXT_PUBLIC_API_BASE_URL}/api/train`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!response.ok) {
        throw new Error("Training failed");
      }

      const modelBlob = await response.blob();

      const downloadUrl = window.URL.createObjectURL(modelBlob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `${formData.get("modelName")}.pkl`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      setAlertMessage({
        type: "success",
        title: "Model Trained Successfully",
        message: `Model "${formData.get("modelName")}" has been trained and is ready for download.`,
      });
    } catch (error) {
      console.error("Error:", error);
      setAlertMessage({
        type: "error",
        title: "Training Failed",
        message:
          "An error occurred while training the model. Please try again.",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadSample = (fileType: "csv" | "json") => {
    const link = document.createElement("a");
    link.href = `/data/sample_data.${fileType}`;
    link.download = `/data/sample_data.${fileType}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="space-y-2">
        <Label htmlFor="file-upload" className="text-sm font-medium">
          Upload File (CSV or JSON)
        </Label>

        <Input
          id="file-upload"
          type="file"
          accept=".csv,.json"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
          required
          className="cursor-pointer"
        />

        <div className="flex gap-2 mt-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="flex-1"
            onClick={() => handleDownloadSample("csv")}
          >
            <Download className="w-4 h-4 mr-2" />
            Sample CSV
          </Button>

          <Button
            type="button"
            variant="outline"
            size="sm"
            className="flex-1"
            onClick={() => handleDownloadSample("json")}
          >
            <Download className="w-4 h-4 mr-2" />
            Sample JSON
          </Button>
        </div>
      </div>

      <div className="space-y-2">
        <Label htmlFor="model-name" className="text-sm font-medium">
          Model Name
        </Label>

        <Input
          id="model-name"
          type="text"
          placeholder="Enter a name for your model"
          value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          required
          className="focus:ring-2 focus:ring-primary"
        />
      </div>

      <Button
        type="submit"
        className="w-full transition-colors duration-200"
        disabled={loading}
      >
        {loading ? "Training..." : "Train Model"}
      </Button>
    </form>
  );
}
