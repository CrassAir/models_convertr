import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';

import 'package:flutter/foundation.dart';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image/image.dart' as img;
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

class OCRService {
  late Interpreter _interpreter;
  final double _confThreshold = 0.3;
  final double _iouThreshold = 0.45;
  late final TextRecognizer _textRecognizer;

  static const Map<int, String> classNames = {
    0: 'cargo_name',
    1: 'cargo_weight',
    2: 'count',
    3: 'date',
    4: 'document_number',
    5: 'receiver',
    6: 'sender',
    7: 'stamp',
  };

  Future<void> init() async {
    _interpreter = await Interpreter.fromAsset('assets/tf_models/best_float32.tflite', options: InterpreterOptions()..threads = 4);
    _textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);
    debugPrint('OCR Service initialized');
  }

  Future<void> dispose() async {
    try {
      _interpreter.close();
      _textRecognizer.close();
    } catch (_) {}
  }

  Future<WaybillData> detectAndOcr(Uint8List bytes) async {
    var original = img.decodeImage(bytes);
    if (original == null) throw StateError('Failed to decode image');

    if (original.width > 2048 || original.height > 2048) {
      final scale = 2048 / max(original.width, original.height);
      original = img.copyResize(original, width: (original.width * scale).round(), height: (original.height * scale).round());
    }

    final candidates = [original, img.copyRotate(original, angle: 180)];

    img.Image best = original;
    double bestScore = -1;
    List<Detection> bestDetections = [];

    for (final test in candidates) {
      final dets = await _runDetection(test);
      final score = dets.fold<double>(0, (s, d) => s + d.confidence);
      if (score > bestScore && dets.isNotEmpty) {
        bestScore = score;
        best = test;
        bestDetections = dets;
      }
    }

    if (bestDetections.isEmpty) {
      return WaybillData.empty();
    }

    final extracted = <String, String>{};
    for (final det in bestDetections) {
      try {
        final rect = det.rect;

        const paddingWidth = 0.20; // 20% расширение по ширине
        const paddingHeight = 0.05; // 5% расширение по высоте

        final expandW = rect.width * paddingWidth;
        final expandH = rect.height * paddingHeight;

        final x = max(0, (rect.left - expandW).floor());
        final y = max(0, (rect.top - expandH).floor());
        final w = min((rect.width + expandW * 2).ceil(), best.width - x);
        final h = min((rect.height + expandH * 2).ceil(), best.height - y);

        if (w <= 10 || h <= 10) continue;

        img.Image crop = img.copyCrop(best, x: x, y: y, width: w, height: h);

        const minWidth = 400;
        const minHeight = 150;
        const maxScale = 4;

        if (crop.width < minWidth || crop.height < minHeight) {
          final scaleW = minWidth / crop.width;
          final scaleH = minHeight / crop.height;
          final scale = min(max(scaleW, scaleH).ceil(), maxScale);

          crop = img.copyResize(crop, width: crop.width * scale, height: crop.height * scale, interpolation: img.Interpolation.cubic);
          debugPrint('Upscaled ${det.className} to ${crop.width}x${crop.height} (${scale}x)');
        }

        try {
          crop = img.grayscale(crop);
          crop = img.contrast(crop, contrast: 160);

          if (crop.width > 10 && crop.height > 10) {
            crop = img.gaussianBlur(crop, radius: 1);
          }

          crop = img.adjustColor(crop, contrast: 1.3, brightness: 1.12);

          if (crop.width > 3 && crop.height > 3) {
            crop = img.convolution(crop, filter: [0, -1, 0, -1, 5, -1, 0, -1, 0]);
          }
        } catch (e) {
          debugPrint('Preprocessing error for ${det.className}: $e');
          continue;
        }

        String text = '';
        try {
          text = await _recognizeText(crop);
        } catch (e) {
          debugPrint('ML Kit error for ${det.className}: $e');
          continue;
        }

        if (text.isEmpty) {
          debugPrint('${det.className}: empty text');
          continue;
        }

        String norm = '';
        try {
          norm = _normalizeText(text, det.className);
        } catch (e) {
          debugPrint('Normalize error for ${det.className}: $e, text: "$text"');
          norm = text.trim();
        }

        if (norm.isNotEmpty) {
          final prev = extracted[det.className];
          if (prev == null || norm.length > prev.length) {
            extracted[det.className] = norm;
            debugPrint('${det.className}: "$norm" (conf: ${det.confidence.toStringAsFixed(2)})');
          }
        }
      } catch (e, stack) {
        debugPrint('OCR error for ${det.className}: $e');
        debugPrint('Stack: $stack');
      }
    }

    return WaybillData.fromMap(extracted);
  }

  Future<(Uint8List, WaybillData)> detectAndVisualize(Uint8List bytes) async {
    var original = img.decodeImage(bytes);
    if (original == null) throw StateError('Failed to decode image');

    if (original.width > 2048 || original.height > 2048) {
      final scale = 2048 / max(original.width, original.height);
      original = img.copyResize(original, width: (original.width * scale).round(), height: (original.height * scale).round());
    }

    final candidates = [original, img.copyRotate(original, angle: 180)];
    img.Image best = original;
    double bestScore = -1;
    List<Detection> bestDetections = [];

    for (final test in candidates) {
      final dets = await _runDetection(test);
      final score = dets.fold<double>(0, (s, d) => s + d.confidence);
      if (score > bestScore && dets.isNotEmpty) {
        bestScore = score;
        best = test;
        bestDetections = dets;
      }
    }

    final extracted = <String, String>{};
    for (final det in bestDetections) {
      try {
        final rect = det.rect;

        const paddingWidth = 0.10; // 10% расширение по ширине
        const paddingHeight = 0.01; // 1% расширение по высоте

        final expandW = rect.width * paddingWidth;
        final expandH = rect.height * paddingHeight;

        final int x = max(0, (rect.left - expandW).floor());
        final int y = max(0, (rect.top - expandH).floor());
        final int w = min((rect.width + expandW * 2).ceil(), best.width - x);
        final int h = min((rect.height + expandH * 2).ceil(), best.height - y);

        det.rect = Rect.fromLTWH(x.toDouble(), y.toDouble(), w.toDouble(), h.toDouble());

        if (w <= 10 || h <= 10) continue;

        img.Image crop = img.copyCrop(best, x: x, y: y, width: w, height: h);

        const minWidth = 400;
        const minHeight = 150;
        const maxScale = 4;

        if (crop.width < minWidth || crop.height < minHeight) {
          final scaleW = minWidth / crop.width;
          final scaleH = minHeight / crop.height;
          final scale = min(max(scaleW, scaleH).ceil(), maxScale);

          crop = img.copyResize(crop, width: crop.width * scale, height: crop.height * scale, interpolation: img.Interpolation.cubic);
          debugPrint('Upscaled ${det.className} to ${crop.width}x${crop.height} (${scale}x)');
        }

        try {
          crop = img.grayscale(crop);
          crop = img.contrast(crop, contrast: 160);

          if (crop.width > 10 && crop.height > 10) {
            crop = img.gaussianBlur(crop, radius: 1);
          }

          crop = img.adjustColor(crop, contrast: 1.3, brightness: 1.12);

          if (crop.width > 3 && crop.height > 3) {
            crop = img.convolution(crop, filter: [0, -1, 0, -1, 5, -1, 0, -1, 0]);
          }
        } catch (e) {
          debugPrint('Preprocessing error for ${det.className}: $e');
          continue;
        }

        String text = '';
        try {
          text = await _recognizeText(crop);
        } catch (e) {
          debugPrint('ML Kit error for ${det.className}: $e');
          continue;
        }

        if (text.isEmpty) {
          debugPrint('${det.className}: empty text');
          continue;
        }

        String norm = '';
        try {
          norm = _normalizeText(text, det.className);
        } catch (e) {
          debugPrint('Normalize error for ${det.className}: $e, text: "$text"');
          norm = text.trim();
        }

        if (norm.isNotEmpty) {
          final prev = extracted[det.className];
          if (prev == null || norm.length > prev.length) {
            extracted[det.className] = norm;
            debugPrint('${det.className}: "$norm" (conf: ${det.confidence.toStringAsFixed(2)})');
          }
        }
      } catch (e, stack) {
        debugPrint('OCR error for ${det.className}: $e');
        debugPrint('Stack: $stack');
      }
    }

    final visualized = _drawDetections(best, bestDetections);

    return (Uint8List.fromList(img.encodeJpg(visualized, quality: 90)),  WaybillData.fromMap(extracted));
  }

  img.Image _drawDetections(img.Image src, List<Detection> detections) {
    final result = img.Image.from(src);

    const classColors = {
      'cargo_name': [255, 0, 0], // Красный
      'cargo_weight': [0, 255, 0], // Зелёный
      'count': [0, 0, 255], // Синий
      'date': [255, 255, 0], // Жёлтый
      'document_number': [255, 0, 255], // Пурпурный
      'receiver': [0, 255, 255], // Циан
      'sender': [255, 128, 0], // Оранжевый
      'stamp': [128, 0, 255], // Фиолетовый
    };

    for (final det in detections) {
      final rect = det.rect;
      final color = classColors[det.className] ?? [255, 255, 255];

      img.drawRect(
        result,
        x1: rect.left.toInt(),
        y1: rect.top.toInt(),
        x2: rect.right.toInt(),
        y2: rect.bottom.toInt(),
        color: img.ColorRgb8(color[0], color[1], color[2]),
        thickness: 3,
      );

      final labelText = '${det.className} ${(det.confidence * 100).toStringAsFixed(0)}%';
      final labelY = max(0, rect.top.toInt() - 20);

      img.fillRect(
        result,
        x1: rect.left.toInt(),
        y1: labelY,
        x2: (rect.left + 200).toInt(),
        y2: rect.top.toInt(),
        color: img.ColorRgb8(color[0], color[1], color[2]),
      );

      img.drawString(result, labelText, font: img.arial24, x: rect.left.toInt() + 5, y: labelY + 2, color: img.ColorRgb8(255, 255, 255));
    }

    return result;
  }

  Future<List<Detection>> _runDetection(img.Image src) async {
    final resized = img.copyResize(src, width: 1024, height: 1024);
    final input = List.generate(1, (_) => List.generate(1024, (_) => List.generate(1024, (_) => List.filled(3, 0.0))));

    for (int y = 0; y < 1024; y++) {
      for (int x = 0; x < 1024; x++) {
        final pixel = resized.getPixel(x, y);
        input[0][y][x][0] = pixel.r / 255.0;
        input[0][y][x][1] = pixel.g / 255.0;
        input[0][y][x][2] = pixel.b / 255.0;
      }
    }

    final output = List.generate(1, (_) => List.generate(12, (_) => List.filled(21504, 0.0)));

    _interpreter.run(input, output);
    return _parseYOLO11Output(output[0], src.width, src.height);
  }

  List<Detection> _parseYOLO11Output(List<dynamic> raw, int srcW, int srcH) {
    final detections = <Detection>[];
    final numBoxes = (raw[0] as List).length;

    for (int i = 0; i < numBoxes; i++) {
      final cx = (raw[0][i] as num).toDouble();
      final cy = (raw[1][i] as num).toDouble();
      final w = (raw[2][i] as num).toDouble();
      final h = (raw[3][i] as num).toDouble();

      double maxConf = 0;
      int maxCls = 0;
      for (int c = 0; c < 8; c++) {
        final conf = (raw[4 + c][i] as num).toDouble();
        if (conf > maxConf) {
          maxConf = conf;
          maxCls = c;
        }
      }

      if (maxConf < _confThreshold) continue;

      final xmin = ((cx - w / 2) * srcW).clamp(0.0, srcW.toDouble());
      final ymin = ((cy - h / 2) * srcH).clamp(0.0, srcH.toDouble());
      final xmax = ((cx + w / 2) * srcW).clamp(0.0, srcW.toDouble());
      final ymax = ((cy + h / 2) * srcH).clamp(0.0, srcH.toDouble());

      final boxW = xmax - xmin;
      final boxH = ymax - ymin;
      if (boxW < 10 || boxH < 10) continue;

      detections.add(
        Detection(
          rect: Rect.fromLTWH(xmin, ymin, boxW, boxH),
          confidence: maxConf,
          classId: maxCls,
          className: classNames[maxCls] ?? 'unknown',
        ),
      );
    }

    return _nms(detections);
  }

  List<Detection> _nms(List<Detection> dets) {
    dets.sort((a, b) => b.confidence.compareTo(a.confidence));
    final keep = <Detection>[];
    while (dets.isNotEmpty) {
      final best = dets.removeAt(0);
      keep.add(best);
      dets.removeWhere((d) => d.classId == best.classId && _iou(best.rect, d.rect) > _iouThreshold);
    }
    return keep;
  }

  double _iou(Rect a, Rect b) {
    final x1 = max(a.left, b.left);
    final y1 = max(a.top, b.top);
    final x2 = min(a.right, b.right);
    final y2 = min(a.bottom, b.bottom);
    if (x2 < x1 || y2 < y1) return 0;
    final inter = (x2 - x1) * (y2 - y1);
    final union = a.width * a.height + b.width * b.height - inter;
    return inter / union;
  }

  Future<String> _recognizeText(img.Image crop) async {
    final bytes = Uint8List.fromList(img.encodeJpg(crop, quality: 95));
    final dir = await getTemporaryDirectory();
    final path = '${dir.path}/crop_${DateTime.now().millisecondsSinceEpoch}.jpg';
    final file = File(path);
    await file.writeAsBytes(bytes);

    try {
      final inputImage = InputImage.fromFilePath(path);
      final result = await _textRecognizer.processImage(inputImage);
      return result.text.trim();
    } finally {
      try {
        if (await file.exists()) await file.delete();
      } catch (_) {}
    }
  }

  String _normalizeText(String text, String field) {
    var t = text.replaceAll('\n', ' ').replaceAll(RegExp(r'\s+'), ' ').trim();

    switch (field) {
      case 'date':
        final m = RegExp(r'(\d{2})[.\/-](\d{2})[.\/-](\d{4})').firstMatch(t);
        if (m != null) {
          return '${m.group(1)}.${m.group(2)}.${m.group(3)}';
        }
        return t;

      case 'document_number':
        return t.replaceAll(RegExp(r'[^\d]'), '');

      case 'cargo_weight':
        final m = RegExp(r'([\d.,]+)\s*(кг|kg|т|t)', caseSensitive: false).firstMatch(t);
        if (m != null) {
          final num = m.group(1)!.replaceAll(',', '.');
          return '$num т';
        }
        return t;

      case 'count':
        final m = RegExp(r'\d+').firstMatch(t);
        return m?.group(0) ?? '0';

      case 'sender':
      case 'receiver':
        t = t.replaceFirst(RegExp(r'^\d+\.\s*'), '');
        t = _fixAllLatinToCyrillic(t);
        t = t.replaceAll(RegExp(r'[^\wА-Яа-я0-9\s\-.,()]', unicode: true), ' ');
        t = t.replaceAll(RegExp(r'\s+'), ' ').trim();
        return t.toLowerCase();

      case 'cargo_name':
        t = t.replaceFirst(RegExp(r'^\d+\.\s*'), '');
        t = _fixAllLatinToCyrillic(t);
        t = t.replaceAll(RegExp(r'[^\wА-Яа-я0-9\s\-.,xх×()]', unicode: true), ' ');
        t = t.replaceAll(RegExp(r'\s+'), ' ').trim();
        var t_list = t.split(' ');
        return t_list.getRange(1, t_list.length - 1).join(' ').toLowerCase();

      case 'stamp':
        t = _fixAllLatinToCyrillic(t);
        t = t.replaceAll(RegExp(r'[^\wА-Яа-я0-9\s\-.,()]', unicode: true), ' ');
        return t.toLowerCase().trim();

      default:
        return t.toLowerCase();
    }
  }

  String _fixAllLatinToCyrillic(String text) {
    const replacements = {
      'A': 'А',
      'a': 'а',
      'B': 'В',
      'b': 'б',
      'E': 'Е',
      'e': 'е',
      'K': 'К',
      'k': 'к',
      'M': 'М',
      'm': 'м',
      'H': 'Н',
      'h': 'н',
      'O': 'О',
      'o': 'о',
      'P': 'Р',
      'p': 'р',
      'C': 'С',
      'c': 'с',
      'T': 'Т',
      't': 'т',
      'Y': 'У',
      'y': 'у',
      'X': 'Х',
      'x': 'х',
      '0': 'о',
    };

    var result = text;
    replacements.forEach((lat, cyr) {
      result = result.replaceAll(lat, cyr);
    });

    result = result
        .replaceAll(RegExp(r'[Il1]', caseSensitive: false), 'і')
        .replaceAll(RegExp(r'0(?=[а-яА-Я])', unicode: true), 'о') // 0 → о перед кириллицей
        .replaceAll('6', 'б')
        .replaceAll('9', 'д');

    return result;
  }
}

class Detection {
  Rect rect;
  final double confidence;
  final int classId;
  final String className;

  Detection({required this.rect, required this.confidence, required this.classId, required this.className});
}

class WaybillData {
  final String? cargoName;
  final String? cargoWeight;
  final String? count;
  final String? date;
  final String? documentNumber;
  final String? receiver;
  final String? sender;
  final String? stamp;

  WaybillData({this.cargoName, this.cargoWeight, this.count, this.date, this.documentNumber, this.receiver, this.sender, this.stamp});

  factory WaybillData.fromMap(Map<String, String> data) {
    return WaybillData(
      cargoName: data['cargo_name'],
      cargoWeight: data['cargo_weight'],
      count: data['count'],
      date: data['date'],
      documentNumber: data['document_number'],
      receiver: data['receiver'],
      sender: data['sender'],
      stamp: data['stamp'],
    );
  }

  factory WaybillData.empty() => WaybillData();

  Map<String, dynamic> toJson() => {
    'cargo_name': cargoName,
    'cargo_weight': cargoWeight,
    'count': count,
    'date': date,
    'document_number': documentNumber,
    'receiver': receiver,
    'sender': sender,
    'stamp': stamp,
  };

  String toPrettyJson() => JsonEncoder.withIndent('  ').convert(toJson());

  @override
  String toString() => toPrettyJson();
}
