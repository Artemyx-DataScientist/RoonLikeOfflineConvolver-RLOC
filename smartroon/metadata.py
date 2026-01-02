from __future__ import annotations

import base64
import copy
import logging
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import mutagen
from mutagen.flac import FLAC, Picture
from mutagen.id3 import APIC, ID3, TALB, TCON, TDRC, TIT2, TPE1, TRCK
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4, MP4Cover
from mutagen.oggopus import OggOpus
from mutagen.oggvorbis import OggVorbis
from mutagen.wave import WAVE

CoverInfo = Tuple[str, bytes]
TagMapping = Dict[str, str]


def _log_replaygain_notice(tag_keys: Iterable[str], logger: logging.Logger) -> None:
    lowered = {key.lower() for key in tag_keys}
    if any(key.startswith("replaygain_") or key.startswith("r128_") for key in lowered):
        logger.debug(
            "Найдены ReplayGain/R128 теги, пересчёт не выполняется и значения копируются как есть."
        )


def _ensure_id3(target: MP3 | WAVE) -> ID3:
    if target.tags is None:
        target.add_tags()
    if not isinstance(target.tags, ID3):
        raise TypeError("Ожидались ID3-теги для MP3/WAV")
    target.tags.clear()
    return target.tags


def _clone_picture(picture: Picture) -> Picture:
    cloned = Picture()
    cloned.data = picture.data
    cloned.mime = picture.mime
    cloned.type = picture.type
    cloned.desc = picture.desc
    cloned.width = picture.width
    cloned.height = picture.height
    cloned.depth = picture.depth
    cloned.colors = picture.colors
    return cloned


def _copy_flac_to_flac(src: FLAC, dst: FLAC, logger: logging.Logger) -> None:
    dst.clear()
    dst.clear_pictures()
    if src.tags is not None:
        _log_replaygain_notice(src.keys(), logger)
        if dst.tags is None:
            dst.add_tags()
        for key, values in src.tags.items():
            dst[key] = list(values)
    for picture in src.pictures:
        dst.add_picture(_clone_picture(picture))
    dst.save()


def _copy_id3_tags(src: ID3, dst: ID3, logger: logging.Logger) -> None:
    _log_replaygain_notice(src.keys(), logger)
    dst.clear()
    for frame in src.values():
        dst.add(copy.copy(frame))


def _copy_mp3_to_mp3(src: MP3, dst: MP3, logger: logging.Logger) -> None:
    dst_tags = _ensure_id3(dst)
    src_tags = src.tags if isinstance(src.tags, ID3) else ID3()
    _copy_id3_tags(src_tags, dst_tags, logger)
    dst.save()


def _copy_ogg_to_ogg(src: OggVorbis | OggOpus, dst: OggVorbis | OggOpus, logger: logging.Logger) -> None:
    if src.tags is None:
        logger.warning("Отсутствуют теги в исходном файле OGG/Opus, копирование пропущено")
        return
    if dst.tags is None:
        dst.add_tags()
    dst.tags.clear()
    _log_replaygain_notice(src.keys(), logger)
    for key, values in src.tags.items():
        dst.tags[key] = list(values)
    pictures = src.tags.get("metadata_block_picture")
    if pictures:
        dst.tags["metadata_block_picture"] = list(pictures)
    dst.save()


def _copy_mp4_to_mp4(src: MP4, dst: MP4, logger: logging.Logger) -> None:
    if dst.tags is None:
        dst.add_tags()
    dst.tags.clear()
    if src.tags:
        _log_replaygain_notice(src.tags.keys(), logger)
        for key, values in src.tags.items():
            dst.tags[key] = copy.deepcopy(values)
    dst.save()


def _copy_wave_to_wave(src: WAVE, dst: WAVE, logger: logging.Logger) -> None:
    if not isinstance(src.tags, ID3):
        logger.warning("WAV метаданные поддерживаются частично: отсутствуют ID3-теги")
        return
    dst_tags = _ensure_id3(dst)
    _copy_id3_tags(src.tags, dst_tags, logger)
    dst.save()


def _extract_common_tags(file: FLAC | MP3 | OggVorbis | OggOpus | MP4 | WAVE) -> TagMapping:
    tags: TagMapping = {}
    if isinstance(file, FLAC) or isinstance(file, (OggVorbis, OggOpus)):
        if not file.tags:
            return tags
        for key in ("artist", "album", "title", "tracknumber", "date", "genre"):
            if key in file and file[key]:
                tags[key] = str(file[key][0])
    elif isinstance(file, MP3):
        id3_tags = file.tags if isinstance(file.tags, ID3) else ID3()
        mapping = {
            "artist": "TPE1",
            "album": "TALB",
            "title": "TIT2",
            "tracknumber": "TRCK",
            "date": "TDRC",
            "genre": "TCON",
        }
        for name, frame_id in mapping.items():
            frame = id3_tags.get(frame_id)
            if frame and getattr(frame, "text", None):
                tags[name] = str(frame.text[0])
    elif isinstance(file, MP4):
        if file.tags:
            mapping = {
                "artist": "\xa9ART",
                "album": "\xa9alb",
                "title": "\xa9nam",
                "date": "\xa9day",
                "genre": "\xa9gen",
            }
            for name, atom in mapping.items():
                values = file.tags.get(atom)
                if values:
                    tags[name] = str(values[0])
            track = file.tags.get("trkn")
            if track:
                track_number = track[0][0]
                if track_number:
                    tags["tracknumber"] = str(track_number)
    elif isinstance(file, WAVE):
        if isinstance(file.tags, ID3):
            mapping = {
                "artist": "TPE1",
                "album": "TALB",
                "title": "TIT2",
                "tracknumber": "TRCK",
                "date": "TDRC",
                "genre": "TCON",
            }
            for name, frame_id in mapping.items():
                frame = file.tags.get(frame_id)
                if frame and getattr(frame, "text", None):
                    tags[name] = str(frame.text[0])
    return tags


def _extract_cover(file: FLAC | MP3 | OggVorbis | OggOpus | MP4 | WAVE) -> CoverInfo | None:
    if isinstance(file, FLAC):
        if file.pictures:
            picture = file.pictures[0]
            return picture.mime, bytes(picture.data)
    elif isinstance(file, (OggVorbis, OggOpus)):
        block_pictures = file.tags.get("metadata_block_picture") if file.tags else None
        if block_pictures:
            picture = Picture()
            picture_data = base64.b64decode(block_pictures[0])
            picture.from_data(picture_data)
            return picture.mime, bytes(picture.data)
    elif isinstance(file, MP3):
        id3_tags = file.tags if isinstance(file.tags, ID3) else ID3()
        for apic in id3_tags.getall("APIC"):
            return apic.mime, bytes(apic.data)
    elif isinstance(file, MP4):
        if file.tags:
            covr = file.tags.get("covr")
            if covr:
                cover = covr[0]
                if isinstance(cover, MP4Cover):
                    mime = "image/png" if cover.imageformat == MP4Cover.FORMAT_PNG else "image/jpeg"
                    return mime, bytes(cover)
    elif isinstance(file, WAVE):
        if isinstance(file.tags, ID3):
            for apic in file.tags.getall("APIC"):
                return apic.mime, bytes(apic.data)
    return None


def _apply_cover_to_flac(dst: FLAC, cover: CoverInfo) -> None:
    picture = Picture()
    picture.data = cover[1]
    picture.mime = cover[0]
    picture.type = 3
    picture.desc = ""
    dst.add_picture(picture)


def _apply_cover_to_ogg(dst: OggVorbis | OggOpus, cover: CoverInfo) -> None:
    picture = Picture()
    picture.data = cover[1]
    picture.mime = cover[0]
    picture.type = 3
    picture.desc = ""
    encoded = base64.b64encode(picture.write()).decode("ascii")
    dst.tags["metadata_block_picture"] = [encoded]


def _apply_cover_to_id3(dst: ID3, cover: CoverInfo) -> None:
    dst.add(APIC(encoding=3, mime=cover[0], type=3, desc="", data=cover[1]))


def _apply_cover_to_mp4(dst: MP4, cover: CoverInfo) -> None:
    image_format = MP4Cover.FORMAT_PNG if cover[0].lower().endswith("png") else MP4Cover.FORMAT_JPEG
    dst.tags["covr"] = [MP4Cover(cover[1], imageformat=image_format)]


def _apply_common_tags_to_flac(dst: FLAC, tags: TagMapping) -> None:
    if dst.tags is None:
        dst.add_tags()
    dst.clear()
    for key, value in tags.items():
        dst[key] = [value]


def _apply_common_tags_to_ogg(dst: OggVorbis | OggOpus, tags: TagMapping) -> None:
    if dst.tags is None:
        dst.add_tags()
    dst.tags.clear()
    for key, value in tags.items():
        dst.tags[key] = [value]


def _apply_common_tags_to_id3(dst: ID3, tags: TagMapping) -> None:
    dst.clear()
    if "artist" in tags:
        dst.add(TPE1(encoding=3, text=[tags["artist"]]))
    if "album" in tags:
        dst.add(TALB(encoding=3, text=[tags["album"]]))
    if "title" in tags:
        dst.add(TIT2(encoding=3, text=[tags["title"]]))
    if "tracknumber" in tags:
        dst.add(TRCK(encoding=3, text=[tags["tracknumber"]]))
    if "date" in tags:
        dst.add(TDRC(encoding=3, text=[tags["date"]]))
    if "genre" in tags:
        dst.add(TCON(encoding=3, text=[tags["genre"]]))


def _apply_common_tags_to_mp4(dst: MP4, tags: TagMapping) -> None:
    if dst.tags is None:
        dst.add_tags()
    dst.tags.clear()
    mapping = {
        "artist": "\xa9ART",
        "album": "\xa9alb",
        "title": "\xa9nam",
        "date": "\xa9day",
        "genre": "\xa9gen",
    }
    for key, atom in mapping.items():
        if key in tags:
            dst.tags[atom] = [tags[key]]
    if "tracknumber" in tags:
        try:
            track_number = int(str(tags["tracknumber"]).split("/")[0])
        except ValueError:
            track_number = None
        if track_number:
            dst.tags["trkn"] = [(track_number, 0)]


def _copy_best_effort(
    src_file: FLAC | MP3 | OggVorbis | OggOpus | MP4 | WAVE,
    dst_file: FLAC | MP3 | OggVorbis | OggOpus | MP4 | WAVE,
    logger: logging.Logger,
) -> None:
    tags = _extract_common_tags(src_file)
    cover = _extract_cover(src_file)
    logger.info(
        "Форматы различаются (%s -> %s), выполняется best-effort перенос тегов.",
        type(src_file).__name__,
        type(dst_file).__name__,
    )
    if isinstance(dst_file, FLAC):
        _apply_common_tags_to_flac(dst_file, tags)
        if cover:
            _apply_cover_to_flac(dst_file, cover)
    elif isinstance(dst_file, (OggVorbis, OggOpus)):
        _apply_common_tags_to_ogg(dst_file, tags)
        if cover:
            _apply_cover_to_ogg(dst_file, cover)
    elif isinstance(dst_file, MP3):
        dst_tags = _ensure_id3(dst_file)
        _apply_common_tags_to_id3(dst_tags, tags)
        if cover:
            _apply_cover_to_id3(dst_tags, cover)
    elif isinstance(dst_file, MP4):
        _apply_common_tags_to_mp4(dst_file, tags)
        if cover and dst_file.tags is not None:
            _apply_cover_to_mp4(dst_file, cover)
    elif isinstance(dst_file, WAVE):
        dst_tags = _ensure_id3(dst_file)
        _apply_common_tags_to_id3(dst_tags, tags)
        if cover:
            _apply_cover_to_id3(dst_tags, cover)
    else:
        raise TypeError("Формат назначения не поддерживается для переноса метаданных")
    dst_file.save()


def copy_metadata(
    src_path: Path | str,
    dst_path: Path | str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Копирует теги и обложки из src в dst.
    Поддерживает: FLAC, WAV (где возможно), MP3, OGG/Vorbis, Opus, M4A/ALAC.
    Если формат не поддерживается или что-то не получилось — логируем warning, но не падаем.
    """

    log = logger or logging.getLogger(__name__)
    source = Path(src_path)
    destination = Path(dst_path)
    try:
        src_file = mutagen.File(source, easy=False)
        dst_file = mutagen.File(destination, easy=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось перенести метаданные для %s: %s", source.name, exc)
        return

    if src_file is None or dst_file is None:
        log.warning(
            "Не удалось перенести метаданные для %s: неподдерживаемый формат источника/назначения",
            source.name,
        )
        return

    try:
        if isinstance(src_file, FLAC) and isinstance(dst_file, FLAC):
            _copy_flac_to_flac(src_file, dst_file, log)
        elif isinstance(src_file, MP3) and isinstance(dst_file, MP3):
            _copy_mp3_to_mp3(src_file, dst_file, log)
        elif isinstance(src_file, (OggVorbis, OggOpus)) and isinstance(
            dst_file, (OggVorbis, OggOpus)
        ):
            _copy_ogg_to_ogg(src_file, dst_file, log)
        elif isinstance(src_file, MP4) and isinstance(dst_file, MP4):
            _copy_mp4_to_mp4(src_file, dst_file, log)
        elif isinstance(src_file, WAVE) and isinstance(dst_file, WAVE):
            _copy_wave_to_wave(src_file, dst_file, log)
        else:
            _copy_best_effort(src_file, dst_file, log)
    except Exception as exc:  # noqa: BLE001
        log.warning("Не удалось перенести метаданные для %s: %s", source.name, exc)
