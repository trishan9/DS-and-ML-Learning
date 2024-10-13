import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { FileUp, AlertCircle, CheckCircle2 } from "lucide-react";
import TrainModelAccordion from "./TrainModelAccordion";
import TrainModelForm from "./TrainModelForm";

interface TrainModelProps {
  isOpen: boolean;
  onClose: () => void;
}

export interface AlertMessage {
  type: "error" | "success";
  title: string;
  message: string;
}

export function TrainModel({ isOpen, onClose }: TrainModelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState("");
  const [alertMessage, setAlertMessage] = useState<AlertMessage | null>(null);

  const handleClose = () => {
    setFile(null);
    setModelName("");
    setAlertMessage(null);
    onClose();
  };

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="sm:max-w-[455px] p-4 md:p-6">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2 text-xl md:text-2xl">
            <FileUp className="w-6 h-6" />
            Train New Model
          </DialogTitle>

          <DialogDescription className="text-sm md:text-base">
            Upload a CSV or JSON file to train a new model.
          </DialogDescription>
        </DialogHeader>

        <TrainModelForm
          file={file}
          modelName={modelName}
          setModelName={setModelName}
          setAlertMessage={setAlertMessage}
          setFile={setFile}
        />

        {alertMessage && (
          <Alert
            variant={alertMessage.type === "error" ? "destructive" : "default"}
            className="mt-4"
          >
            {alertMessage.type === "error" ? (
              <AlertCircle className="h-4 w-4" />
            ) : (
              <CheckCircle2 className="h-4 w-4" />
            )}
            <AlertTitle className="text-sm font-semibold">
              {alertMessage.title}
            </AlertTitle>

            <AlertDescription className="text-sm">
              {alertMessage.message}
            </AlertDescription>
          </Alert>
        )}

        <DialogFooter>
          <TrainModelAccordion />
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
