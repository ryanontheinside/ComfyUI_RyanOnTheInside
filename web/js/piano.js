export function drawPiano(ctx, node) {
    const margin = 10;
    const visibleWidth = node.size[0] - 2 * margin;
    const totalWidth = 1000; // Fixed total width of the piano
    const height = 60;
    const y = node.size[1] - height - margin;

    ctx.save();
    ctx.beginPath();
    ctx.rect(margin, y, visibleWidth, height);
    ctx.clip();

    ctx.fillStyle = "rgba(0, 0, 255, 0.1)";
    ctx.fillRect(margin, y, totalWidth, height);

    const whiteKeyWidth = totalWidth / 52;
    const blackKeyWidth = whiteKeyWidth * 0.65;
    const blackKeyHeight = height * 0.6;
    let xPos = margin - (node.pianoScroll || 0);

    // Draw white keys
    for (let i = 0; i < 88; i++) {
        const midiNote = i + 21;
        const noteInOctave = (midiNote - 12) % 12;
        const isBlack = [1, 3, 6, 8, 10].includes(noteInOctave);
        const isAvailable = node.availableNotes.has(midiNote);

        if (!isBlack) {
            ctx.fillStyle = isAvailable ? "white" : "lightgray";
            ctx.fillRect(xPos, y, whiteKeyWidth, height);
            ctx.strokeStyle = "black";
            ctx.strokeRect(xPos, y, whiteKeyWidth, height);

            if (node.properties.selectedNotes && node.properties.selectedNotes.includes(midiNote)) {
                ctx.fillStyle = "green";
                ctx.fillRect(xPos + 2, y + height - 10, whiteKeyWidth - 4, 8);
            }

            xPos += whiteKeyWidth;
        }
    }

    // Draw black keys
    xPos = margin - (node.pianoScroll || 0);
    for (let i = 0; i < 88; i++) {
        const midiNote = i + 21;
        const noteInOctave = (midiNote - 12) % 12;
        const isBlack = [1, 3, 6, 8, 10].includes(noteInOctave);
        const isAvailable = node.availableNotes.has(midiNote);

        if (isBlack) {
            ctx.fillStyle = isAvailable ? "black" : "darkgray";
            ctx.fillRect(xPos - blackKeyWidth / 2, y, blackKeyWidth, blackKeyHeight);

            if (node.properties.selectedNotes && node.properties.selectedNotes.includes(midiNote)) {
                ctx.fillStyle = "green";
                ctx.fillRect(xPos - blackKeyWidth / 2 + 2, y + blackKeyHeight - 10, blackKeyWidth - 4, 8);
            }
        } else {
            xPos += whiteKeyWidth;
        }
    }

    drawScrollIndicators(ctx, node, margin, y, visibleWidth, totalWidth, height);

    ctx.restore();
}

export function drawScrollIndicators(ctx, node, margin, y, visibleWidth, totalWidth, height) {
    if (node.pianoScroll > 0) {
        const indicatorWidth = 20;
        const indicatorHeight = height;
        ctx.fillStyle = "black";
        ctx.fillRect(margin, y, indicatorWidth, indicatorHeight);

        ctx.fillStyle = "rgba(50, 50, 50, 0.95)";
        ctx.beginPath();
        ctx.moveTo(margin + 10, y + 15);
        ctx.lineTo(margin + 5, y + indicatorHeight / 2);
        ctx.lineTo(margin + 10, y + indicatorHeight - 15);
        ctx.fill();

        node.scrollLeftIndicator = { x: margin, y, width: indicatorWidth, height: indicatorHeight };
    }

    if (node.pianoScroll < totalWidth - visibleWidth) {
        const indicatorWidth = 20;
        const indicatorHeight = height;
        ctx.fillStyle = "rgba(10, 10, 10, 1)";
        ctx.fillRect(node.size[0] - margin - indicatorWidth, y, indicatorWidth, indicatorHeight);

        ctx.fillStyle = "rgba(50, 50, 50, 0.95)";
        ctx.beginPath();
        ctx.moveTo(node.size[0] - margin - indicatorWidth + 10, y + 15);
        ctx.lineTo(node.size[0] - margin - 5, y + indicatorHeight / 2);
        ctx.lineTo(node.size[0] - margin - indicatorWidth + 10, y + indicatorHeight - 15);
        ctx.fill();

        node.scrollRightIndicator = { x: node.size[0] - margin - indicatorWidth, y, width: indicatorWidth, height: indicatorHeight };
    }
}

export function handlePianoMouseDown(node, x, y) {
    const margin = 10;
    const visibleWidth = node.size[0] - 2 * margin;
    const totalWidth = 1000;
    const height = 60;
    const pianoY = node.size[1] - height - margin;

    if (node.scrollLeftIndicator && x >= node.scrollLeftIndicator.x && x <= node.scrollLeftIndicator.x + node.scrollLeftIndicator.width &&
        y >= node.scrollLeftIndicator.y && y <= node.scrollLeftIndicator.y + node.scrollLeftIndicator.height) {
        node.pianoScroll = Math.max(0, (node.pianoScroll || 0) - 20);
        return true;
    }

    if (node.scrollRightIndicator && x >= node.scrollRightIndicator.x && x <= node.scrollRightIndicator.x + node.scrollRightIndicator.width &&
        y >= node.scrollRightIndicator.y && y <= node.scrollRightIndicator.y + node.scrollRightIndicator.height) {
        const maxScroll = totalWidth - visibleWidth;
        node.pianoScroll = Math.min(maxScroll, (node.pianoScroll || 0) + 20);
        return true;
    }

    if (y >= pianoY && y <= pianoY + height) {
        const whiteKeyWidth = totalWidth / 52;
        const blackKeyWidth = whiteKeyWidth * 0.65;
        const blackKeyHeight = height * 0.6;
        let xPos = margin - (node.pianoScroll || 0);
        let clickedNote = null;

        for (let i = 0; i < 88; i++) {
            const midiNote = i + 21;
            const noteInOctave = (midiNote - 12) % 12;
            const isBlack = [1, 3, 6, 8, 10].includes(noteInOctave);

            if (isBlack) {
                if (x >= xPos - blackKeyWidth / 2 && x < xPos + blackKeyWidth / 2 && y <= pianoY + blackKeyHeight) {
                    clickedNote = midiNote;
                    break;
                }
            } else {
                xPos += whiteKeyWidth;
            }
        }

        if (clickedNote === null) {
            xPos = margin - (node.pianoScroll || 0);
            for (let i = 0; i < 88; i++) {
                const midiNote = i + 21;
                const noteInOctave = (midiNote - 12) % 12;
                const isBlack = [1, 3, 6, 8, 10].includes(noteInOctave);

                if (!isBlack) {
                    if (x >= xPos && x < xPos + whiteKeyWidth) {
                        clickedNote = midiNote;
                        break;
                    }
                    xPos += whiteKeyWidth;
                }
            }
        }

        if (clickedNote !== null && node.availableNotes.has(clickedNote)) {
            if (!node.properties.selectedNotes) {
                node.properties.selectedNotes = [];
            }
    
            if (node.properties.selectedNotes.includes(clickedNote)) {
                node.properties.selectedNotes = node.properties.selectedNotes.filter(n => n !== clickedNote);
            } else {
                node.properties.selectedNotes.push(clickedNote);
            }
    
            // Use requestAnimationFrame to ensure the update happens in the next frame
            requestAnimationFrame(() => {
                node.syncNotesWidget();
                node.setDirtyCanvas(true, true);
            });
        }
        return true;
    }

    return false;
}