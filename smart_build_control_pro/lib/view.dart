import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:smart_build_control_pro/services/ocr3.dart';
import 'dart:typed_data';

class DetectionVisualizerPage extends StatefulWidget {
  const DetectionVisualizerPage({super.key});

  @override
  State<DetectionVisualizerPage> createState() => _DetectionVisualizerPageState();
}

class _DetectionVisualizerPageState extends State<DetectionVisualizerPage> {
  final OCRService _ocrService = OCRService();
  Uint8List? _visualizedImage;
  WaybillData? waybillData;
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _ocrService.init();
  }

  @override
  void dispose() {
    _ocrService.dispose();
    super.dispose();
  }

  Future<void> _pickAndVisualize() async {
    final picker = ImagePicker();
    final image = await picker.pickImage(source: ImageSource.gallery);

    if (image == null) return;

    setState(() => _isLoading = true);

    try {
      final bytes = await image.readAsBytes();
      final data = await _ocrService.detectAndVisualize(bytes);
      setState(() {
        waybillData = data.$2;
        _visualizedImage = data.$1;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Ошибка: $e')));
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Визуализация детекций'),
        actions: [IconButton(icon: Icon(Icons.photo_library), onPressed: _pickAndVisualize)],
      ),
      body: _isLoading
          ? Center(child: CircularProgressIndicator())
          : _visualizedImage == null
          ? Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.image, size: 64, color: Colors.grey),
                  SizedBox(height: 16),
                  Text('Выберите изображение для визуализации'),
                  SizedBox(height: 16),
                  ElevatedButton.icon(onPressed: _pickAndVisualize, icon: Icon(Icons.upload_file), label: Text('Выбрать фото')),
                ],
              ),
            )
          : ListView(
              children: [
                InteractiveViewer(minScale: 0.5, maxScale: 4.0, child: Center(child: Image.memory(_visualizedImage!))),
                if (waybillData != null) Padding(padding: const EdgeInsets.all(32), child: Column(
                  children: [
                    Text('Document number: ${waybillData?.documentNumber}'),
                    Text('Date: ${waybillData?.date}'),
                    Text('Cargo name: ${waybillData?.cargoName}'),
                    Text('Cargo weight: ${waybillData?.cargoWeight}'),
                    Text('Count: ${waybillData?.count}'),
                    Text('Sender: ${waybillData?.sender}'),
                    Text('Receiver: ${waybillData?.receiver}'),
                  ],
                )),
              ],
            ),
      floatingActionButton: _visualizedImage != null
          ? FloatingActionButton(onPressed: _pickAndVisualize, child: Icon(Icons.refresh), tooltip: 'Выбрать другое фото')
          : null,
    );
  }
}
