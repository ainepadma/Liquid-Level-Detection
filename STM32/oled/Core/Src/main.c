/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "adc.h"
#include "i2c.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "OLED.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define RX_BUF_SIZE 128
#define MEDIAN_SAMPLES 5

/* 标定相关 */
#define CALIB_SAMPLE_COUNT 20 // 标定采集帧数
#define H_REAL 198.0f         // 标定用瓶子真实高度 (mm)
#define SKIP_FRAMES_CAL 3     // 标定时舍弃的开始帧数

/* 测量相关 */
#define MEASURE_SAMPLE_CNT 20   // 采集有效帧数
#define MEASURE_TIMEOUT_MS 3000 // 3 秒超时
#define SKIP_FRAMES_MEAS 3      // 测量时舍弃的开始帧数

/* 镜头前端到传感器距离 (mm) */
#define LENS_TO_SENSOR_MM 50.0f

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* 系统工作状态 */
typedef enum
{
    STATE_IDLE,
    STATE_CAL1_COLLECT,
    STATE_CAL1_DONE,
    STATE_CAL2_COLLECT,
    STATE_CAL2_DONE,
    STATE_CAL3_COLLECT,
    STATE_CAL3_DONE,
    STATE_MEASURE
} SystemState;
SystemState state;

/* 标定数据（三个标定点：100cm, 80cm, 60cm） */
float h1_px, h2_px, h3_px;
float t1_px, t2_px, t3_px;
float b1_px, b2_px, b3_px;

/* 测量结果 */
float D_mm;
float H_mm;
float L_mm;
float conf;
int level_px;

/* 电流 / 功率 */
float I;       // 电流 (mA)
float P_inst;   // 瞬时功率 (mW)
float Pmax;    // 最大功率 (mW)

/* 最新测量颜色名称 */
char measured_color[8];

/* 颜色码到颜色名称的映射表（可根据实际协议修改） */
const char *color_names[] = {
    "None",    // 0
    "Red",     // 1
    "Orange",  // 2
    "Yellow",  // 3
    "Green",   // 4
    "Cyan",    // 5
    "Blue",    // 6
    "Purple",  // 7
    "Black"    // 8   ← 新增
};

/* 串口接收相关 */
uint8_t uart_rx_buf[RX_BUF_SIZE];
uint8_t uart_rx_index;
uint8_t uart_frame_ready;
uint8_t rx_byte;

/* 标定数据缓存 */
float calib_buffer_h[CALIB_SAMPLE_COUNT];
float calib_buffer_t[CALIB_SAMPLE_COUNT];
float calib_buffer_b[CALIB_SAMPLE_COUNT];
int calib_count;
int cal_skip_count;

/* 测量相关变量 */
uint8_t measuring;
uint32_t measure_start_time;
uint8_t top_updated;
uint8_t meas_skip_count;

/* 测量数据缓存 */
float D_buf[MEASURE_SAMPLE_CNT];
float H_buf[MEASURE_SAMPLE_CNT];
float L_buf[MEASURE_SAMPLE_CNT];
int measure_cnt;

/* ADC 相关 */
uint16_t adc_raw_value;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* 串口接收中断回调 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        if (rx_byte == '\r')
        {
            HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
            return;
        }
        if (rx_byte == '\n')
        {
            uart_rx_buf[uart_rx_index] = '\0';
            uart_frame_ready = 1;
            uart_rx_index = 0;
        }
        else
        {
            uart_rx_buf[uart_rx_index] = rx_byte;
            uart_rx_index++;
            if (uart_rx_index >= RX_BUF_SIZE - 1)
            {
                uart_rx_buf[uart_rx_index] = '\0';
                uart_frame_ready = 1;
                uart_rx_index = 0;
            }
        }
        HAL_UART_Receive_IT(&huart1, &rx_byte, 1);
    }
}

/* 将颜色码（字符串）转换为颜色名称 */
void convert_color_code_to_name(const char *code_str, char *out_name, size_t out_size)
{
    // 尝试将码解释为数字
    int code = -1;
    if (sscanf(code_str, "%d", &code) == 1)
    {
        // 数字映射
        if (code >= 0 && code < (int)(sizeof(color_names) / sizeof(color_names[0])))
        {
            snprintf(out_name, out_size, "%s", color_names[code]);
        }
        else
        {
            snprintf(out_name, out_size, "?%d", code);
        }
    }
    else
    {
        // 不是数字，不处理

    }
}

/* 解析一帧数据，格式: "h,T,B,conf,color_code,level_dist" */
void parse_data_frame(uint8_t *buffer)
{
    int h_px, T_px, B_px, level_dist_px;
    char color_code_str[8];
    float conf_val;

    int ret = sscanf((char *)buffer, "%d,%d,%d,%f,%7[^,],%d",
                     &h_px, &T_px, &B_px, &conf_val, color_code_str, &level_dist_px);
    if (ret != 6)
        return;

    conf = conf_val;
    level_px = level_dist_px;

    // 将颜色码转换为颜色名称
    convert_color_code_to_name(color_code_str, measured_color, sizeof(measured_color));

    /* ---------- 标定阶段 ---------- */
    if (state == STATE_CAL1_COLLECT || state == STATE_CAL2_COLLECT ||
        state == STATE_CAL3_COLLECT)
    {
        if (cal_skip_count < SKIP_FRAMES_CAL)
        {
            cal_skip_count++;
            return;
        }
        if (h_px > 0 && calib_count < CALIB_SAMPLE_COUNT)
        {
            calib_buffer_h[calib_count] = (float)h_px;
            calib_buffer_t[calib_count] = (float)T_px;
            calib_buffer_b[calib_count] = (float)B_px;
            calib_count++;
        }
    }
    /* ---------- 测量阶段 ---------- */
    else if (state == STATE_MEASURE && measuring && !top_updated)
    {
        if (meas_skip_count < SKIP_FRAMES_MEAS)
        {
            meas_skip_count++;
            return;
        }
        if (level_dist_px != -1 && h_px > 0 && measure_cnt < MEASURE_SAMPLE_CNT)
        {
            // 三组标定数据，距离由远到近：1000mm, 800mm, 600mm
            static const float cal_D[3] = {1000.0f, 800.0f, 600.0f};
            float cal_b[3] = {b1_px, b2_px, b3_px};
            float cal_t[3] = {t1_px, t2_px, t3_px};
            float cal_h[3] = {h1_px, h2_px, h3_px};

            float D_ref, h_ref, t_ref;

            // 根据底部坐标 B_px 分段插值/外推
            if (B_px >= cal_b[2])  // 距离 <= 600mm
            {
                float frac = (B_px - cal_b[2]) / (cal_b[1] - cal_b[2]);
                D_ref = cal_D[2] + frac * (cal_D[1] - cal_D[2]);
                h_ref = cal_h[2] + frac * (cal_h[1] - cal_h[2]);
                t_ref = cal_t[2] + frac * (cal_t[1] - cal_t[2]);
            }
            else if (B_px <= cal_b[0])  // 距离 >= 1000mm
            {
                float frac = (B_px - cal_b[0]) / (cal_b[1] - cal_b[0]);
                D_ref = cal_D[0] + frac * (cal_D[1] - cal_D[0]);
                h_ref = cal_h[0] + frac * (cal_h[1] - cal_h[0]);
                t_ref = cal_t[0] + frac * (cal_t[1] - cal_t[0]);
            }
            else  // 在 600~1000mm 之间
            {
                int i;
                for (i = 0; i < 2; i++)
                {
                    if ((B_px >= cal_b[i+1] && B_px <= cal_b[i]) ||
                        (B_px <= cal_b[i+1] && B_px >= cal_b[i]))
                    {
                        float frac = (B_px - cal_b[i]) / (cal_b[i+1] - cal_b[i]);
                        D_ref = cal_D[i] + frac * (cal_D[i+1] - cal_D[i]);
                        h_ref = cal_h[i] + frac * (cal_h[i+1] - cal_h[i]);
                        t_ref = cal_t[i] + frac * (cal_t[i+1] - cal_t[i]);
                        break;
                    }
                }
            }

            // 修正顶边误差
            float T_error = T_px - t_ref;
            float B_eff = B_px - T_error;

            // 用修正后的 B_eff 重新计算 D_ref
            if (B_eff >= cal_b[2])
            {
                float frac = (B_eff - cal_b[2]) / (cal_b[1] - cal_b[2]);
                D_ref = cal_D[2] + frac * (cal_D[1] - cal_D[2]);
            }
            else if (B_eff <= cal_b[0])
            {
                float frac = (B_eff - cal_b[0]) / (cal_b[1] - cal_b[0]);
                D_ref = cal_D[0] + frac * (cal_D[1] - cal_D[0]);
            }
            else
            {
                int i;
                for (i = 0; i < 2; i++)
                {
                    if ((B_eff >= cal_b[i+1] && B_eff <= cal_b[i]) ||
                        (B_eff <= cal_b[i+1] && B_eff >= cal_b[i]))
                    {
                        float frac = (B_eff - cal_b[i]) / (cal_b[i+1] - cal_b[i]);
                        D_ref = cal_D[i] + frac * (cal_D[i+1] - cal_D[i]);
                        break;
                    }
                }
            }

            // 利用镜头前端到传感器距离计算物距
            float D_ref_sensor = D_ref + LENS_TO_SENSOR_MM;
            float D_sensor = D_ref_sensor * h_ref / h_px;
            float D_tmp = D_sensor - LENS_TO_SENSOR_MM;

            // 高度和液位比例
            float ratio = H_REAL / h_ref * D_ref_sensor / D_sensor;
            float H_tmp = ratio * h_px;
            float L_tmp = ratio * level_dist_px;

            D_buf[measure_cnt] = D_tmp;
            H_buf[measure_cnt] = H_tmp;
            L_buf[measure_cnt] = L_tmp;
            measure_cnt++;
        }
    }
}

// 一阶低通滤波
float low_pass_filter(float new_val)
{
    static float lpf_val;
    lpf_val = lpf_val * 0.95f + new_val * 0.05f;
    return lpf_val;
}

/* 中值滤波读取 ADC（电流） */
uint16_t get_adc_median(void)
{
    uint16_t values[MEDIAN_SAMPLES];
    HAL_ADC_Start(&hadc1);
    for (uint8_t i = 0; i < MEDIAN_SAMPLES; i++)
    {
        if (HAL_ADC_PollForConversion(&hadc1, 10) == HAL_OK)
            values[i] = HAL_ADC_GetValue(&hadc1);
        else
            values[i] = 0;
    }
    HAL_ADC_Stop(&hadc1);

    for (uint8_t i = 0; i < MEDIAN_SAMPLES - 1; i++)
    {
        for (uint8_t j = i + 1; j < MEDIAN_SAMPLES; j++)
        {
            if (values[i] > values[j])
            {
                uint16_t temp = values[i];
                values[i] = values[j];
                values[j] = temp;
            }
        }
    }
    return values[MEDIAN_SAMPLES / 2];
}

/* 读取电流，计算瞬时电流（mA）和最大功率（mW） */
void Read_Current(void)
{
    uint16_t adc_value = get_adc_median();
    adc_raw_value = adc_value;
    float voltage = (adc_value / 4095.0f) * 3.3f;
    voltage = low_pass_filter(voltage);

#define INA240_GAIN 20.0f
#define INA240_VREF 1.63f
#define INA240_RSHUNT 0.1f

    float current = (voltage - INA240_VREF) / (INA240_GAIN * INA240_RSHUNT);
    if (current < -5.0f) current = -5.0f;
    if (current > 5.0f)  current = 5.0f;

    I = current * 1000;
    float power_mW = 5.0f * fabsf(current) * 1000;
    if (power_mW > Pmax) Pmax = power_mW;
    P_inst = power_mW;
}

/* 按键下降沿检测（带消抖） */
uint8_t is_key_pressed(void)
{
    static uint8_t last_state = 1;
    uint8_t current = HAL_GPIO_ReadPin(KEY_START_GPIO_Port, KEY_START_Pin);
    uint8_t pressed = 0;

    if (last_state == 1 && current == 0)
    {
        pressed = 1;
        HAL_Delay(20);
        if (HAL_GPIO_ReadPin(KEY_START_GPIO_Port, KEY_START_Pin) != 0)
            pressed = 0;
    }
    last_state = current;
    return pressed;
}

/* 刷新第三、四行：电流（mA）和功率+颜色 */
void refresh_power_display(void)
{
    char line3[20];
    sprintf(line3, "I%dmA P%dmW", (int)(fabsf(I) + 0.5f), (int)(P_inst + 0.5f));
    OLED_ShowString(3, 1, line3);

    char line4[20];
    sprintf(line4, "Pm%dmW %s", (int)(Pmax + 0.5f), measured_color);
    OLED_ShowString(4, 1, line4);
}

/* 刷新上面两行：D、H、L（整数 mm） */
void update_top_lines(void)
{
    char line1[20];
    sprintf(line1, "D%dmm", (int)(D_mm + 0.5f));
    OLED_ShowString(1, 1, line1);

    char line2[20];
    if (H_mm >= 0 && L_mm >= 0)
        sprintf(line2, "H%dmm L%dmm", (int)(H_mm + 0.5f), (int)(L_mm + 0.5f));
    else
        sprintf(line2, "No Level");
    OLED_ShowString(2, 1, line2);
}

/* 去极值平均（去掉 remove 个最大和最小） */
float filter_array(float *arr, int n, int remove)
{
    if (n <= 2 * remove)
    {
        float sum = 0;
        for (int i = 0; i < n; i++) sum += arr[i];
        return (n > 0) ? (sum / n) : 0.0f;
    }
    for (int i = 0; i < n - 1; i++)
        for (int j = i + 1; j < n; j++)
            if (arr[i] > arr[j])
            {
                float t = arr[i]; arr[i] = arr[j]; arr[j] = t;
            }
    float sum = 0;
    for (int i = remove; i < n - remove; i++) sum += arr[i];
    return sum / (n - 2 * remove);
}

/* USER CODE END 0 */

int main(void)
{
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_I2C1_Init();
    MX_USART1_UART_Init();
    MX_ADC1_Init();

    printf("D-TI System Start\r\n");
    if (HAL_ADCEx_Calibration_Start(&hadc1) != HAL_OK)
        Error_Handler();

    OLED_Init();
    OLED_Clear();
    OLED_ShowString(1, 1, "Press Key");
    OLED_ShowString(2, 1, "Cal at 100cm");
    HAL_UART_Receive_IT(&huart1, &rx_byte, 1);

    while (1)
    {
        if (is_key_pressed())
        {
            switch (state)
            {
                case STATE_IDLE:
                    state = STATE_CAL1_COLLECT;
                    calib_count = 0; cal_skip_count = 0;
                    OLED_Clear();
                    OLED_ShowString(1, 1, "Calibrating...");
                    OLED_ShowString(2, 1, "Cal at 100cm");
                    break;
                case STATE_CAL1_DONE:
                    state = STATE_CAL2_COLLECT;
                    calib_count = 0; cal_skip_count = 0;
                    OLED_Clear();
                    OLED_ShowString(1, 1, "Calibrating...");
                    OLED_ShowString(2, 1, "Cal at 80cm");
                    break;
                case STATE_CAL2_DONE:
                    state = STATE_CAL3_COLLECT;
                    calib_count = 0; cal_skip_count = 0;
                    OLED_Clear();
                    OLED_ShowString(1, 1, "Calibrating...");
                    OLED_ShowString(2, 1, "Cal at 60cm");
                    break;
                case STATE_CAL3_DONE:
                    state = STATE_MEASURE;
                    measuring = 1; measure_cnt = 0; meas_skip_count = 0;
                    top_updated = 0; Pmax = 0.0f;
                    measure_start_time = HAL_GetTick();
                    OLED_Clear();
                    update_top_lines();
                    break;
                case STATE_MEASURE:
                    measuring = !measuring;
                    if (measuring)
                    {
                        measure_cnt = 0; meas_skip_count = 0;
                        top_updated = 0; Pmax = 0.0f;
                        measure_start_time = HAL_GetTick();
                        OLED_Clear();
                        update_top_lines();
                    }
                    else
                    {
                        OLED_Clear();
                        OLED_ShowString(1, 1, "PAUSED");
                    }
                    break;
                default: break;
            }
        }

        if (uart_frame_ready)
        {
            uart_frame_ready = 0;
            parse_data_frame(uart_rx_buf);
            uart_rx_index = 0;
        }

        // 标定完成判定
        if (state == STATE_CAL1_COLLECT && calib_count >= CALIB_SAMPLE_COUNT)
        {
            h1_px = filter_array(calib_buffer_h, CALIB_SAMPLE_COUNT, 3);
            t1_px = filter_array(calib_buffer_t, CALIB_SAMPLE_COUNT, 3);
            b1_px = filter_array(calib_buffer_b, CALIB_SAMPLE_COUNT, 3);
            state = STATE_CAL1_DONE;
            OLED_Clear();
            OLED_ShowString(1, 1, "Cal at 100cm OK");
            OLED_ShowString(2, 1, "Press Key -> 80cm");
        }
        else if (state == STATE_CAL2_COLLECT && calib_count >= CALIB_SAMPLE_COUNT)
        {
            h2_px = filter_array(calib_buffer_h, CALIB_SAMPLE_COUNT, 3);
            t2_px = filter_array(calib_buffer_t, CALIB_SAMPLE_COUNT, 3);
            b2_px = filter_array(calib_buffer_b, CALIB_SAMPLE_COUNT, 3);
            state = STATE_CAL2_DONE;
            OLED_Clear();
            OLED_ShowString(1, 1, "Cal at 80cm OK");
            OLED_ShowString(2, 1, "Press Key -> 60cm");
        }
        else if (state == STATE_CAL3_COLLECT && calib_count >= CALIB_SAMPLE_COUNT)
        {
            h3_px = filter_array(calib_buffer_h, CALIB_SAMPLE_COUNT, 3);
            t3_px = filter_array(calib_buffer_t, CALIB_SAMPLE_COUNT, 3);
            b3_px = filter_array(calib_buffer_b, CALIB_SAMPLE_COUNT, 3);
            state = STATE_CAL3_DONE;
            OLED_Clear();
            OLED_ShowString(1, 1, "Cal at 60cm OK");
            OLED_ShowString(2, 1, "Press to Start");
        }

        // 测量运行时刷新功率和距离
        if (state == STATE_MEASURE && measuring)
        {
            static uint32_t last_power_refresh = 0;
            uint32_t now = HAL_GetTick();
            if (now - last_power_refresh >= 200)
            {
                last_power_refresh = now;
                Read_Current();
                refresh_power_display();
            }

            if (!top_updated)
            {
                if (measure_cnt >= MEASURE_SAMPLE_CNT)
                {
                    D_mm = filter_array(D_buf, MEASURE_SAMPLE_CNT, 2);
                    H_mm = filter_array(H_buf, MEASURE_SAMPLE_CNT, 2);
                    L_mm = filter_array(L_buf, MEASURE_SAMPLE_CNT, 2);
                    update_top_lines();
                    top_updated = 1;
                }
                else if (HAL_GetTick() - measure_start_time > MEASURE_TIMEOUT_MS)
                {
                    if (measure_cnt > 0)
                    {
                        D_mm = filter_array(D_buf, measure_cnt, 2);
                        H_mm = filter_array(H_buf, measure_cnt, 2);
                        L_mm = filter_array(L_buf, measure_cnt, 2);
                    }
                    else
                    {
                        D_mm = 0.0f; H_mm = 0.0f; L_mm = 0.0f;
                    }
                    update_top_lines();
                    top_updated = 1;
                }
            }
        }
        else
        {
            HAL_Delay(50);
        }
        HAL_Delay(1);
    }
}

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
    RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
    RCC_OscInitStruct.HSEState = RCC_HSE_ON;
    RCC_OscInitStruct.HSEPredivValue = RCC_HSE_PREDIV_DIV1;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
    RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL9;
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
        Error_Handler();

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
    RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
        Error_Handler();

    PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_ADC;
    PeriphClkInit.AdcClockSelection = RCC_ADCPCLK2_DIV6;
    if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
        Error_Handler();
}

void Error_Handler(void)
{
    __disable_irq();
    while (1) {}
}
